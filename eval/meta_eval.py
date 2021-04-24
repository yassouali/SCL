from __future__ import print_function

import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm
from torch.nn.utils.weight_norm import WeightNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out

def Proto(support, support_ys, query, opt):
    """Protonet classifier"""
    nc = support.shape[-1]
    support = np.reshape(support, (-1, 1, opt.n_ways, opt.n_shots, nc))
    support = support.mean(axis=3)
    batch_size = support.shape[0]
    query = np.reshape(query, (batch_size, -1, 1, nc))
    logits = - ((query - support)**2).sum(-1)
    pred = np.argmax(logits, axis=-1)
    pred = np.reshape(pred, (-1,))
    return pred

def NN(support, support_ys, query):
    """nearest classifier"""
    support = np.expand_dims(support.transpose(), 0)
    query = np.expand_dims(query, 2)

    diff = np.multiply(query - support, query - support)
    distance = diff.sum(1)
    min_idx = np.argmin(distance, axis=1)
    pred = [support_ys[idx] for idx in min_idx]
    return pred

def Cosine(support, support_ys, query):
    """Cosine classifier"""
    support_norm = np.linalg.norm(support, axis=1, keepdims=True)
    support = support / support_norm
    query_norm = np.linalg.norm(query, axis=1, keepdims=True)
    query = query / query_norm

    cosine_distance = query @ support.transpose()
    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [support_ys[idx] for idx in max_idx]
    return pred



def meta_test_scikit_learn(net, testloader, use_logit=False, is_norm=True, classifier='LR', opt=None, **_):
    """
    meta testing loop using scikit learn implementation of the classifier (linear regression by default)
    """
    net = net.eval()
    acc = []

    with torch.no_grad():
        for idx, data in tqdm(enumerate(testloader), total=len(testloader)):
            # fetch the data
            support_xs, support_ys, query_xs, query_ys = data
            support_xs = support_xs.cuda()
            query_xs = query_xs.cuda()
            
            # reshape the inputs
            batch_size, _, channel, height, width = support_xs.size()
            support_xs = support_xs.view(-1, channel, height, width)
            query_xs = query_xs.view(-1, channel, height, width)

            # reshape the labels
            support_ys = support_ys.view(-1).numpy()
            query_ys = query_ys.view(-1).numpy()
            
            _, spatial_f_s, _, support_features = net(support_xs)
            _, spatial_f_q, _, query_features = net(query_xs)

            # reduce the spatial dimensions (avoiding overfitting)
            spatial_f_s = nn.AdaptiveMaxPool2d(2)(spatial_f_s)
            spatial_f_q = nn.AdaptiveMaxPool2d(2)(spatial_f_q)

            # reshape to B, C_dim (= C * 2 * 2)
            support_features_spatial = spatial_f_s.view(spatial_f_s.size(0), -1)
            query_features_spatial = spatial_f_q.view(spatial_f_q.size(0), -1)

            # l2 normalization of the features
            support_features = normalize(support_features)
            query_features = normalize(query_features)

            support_features_spatial = normalize(support_features_spatial)
            query_features_spatial = normalize(query_features_spatial)

            # convert the tensor to numpy
            support_features = support_features.detach().cpu().numpy()
            query_features = query_features.detach().cpu().numpy()

            support_features_spatial = support_features_spatial.detach().cpu().numpy()
            query_features_spatial = query_features_spatial.detach().cpu().numpy()
            
            # training the base classifier
            scores_spatial, scores_global = 0.0, 0.0
            
            assert opt.use_spatial_feat or opt.use_global_feat, "choose the usage of either spa. of glob. features"
        
            # train on the spatial features
            if opt.use_spatial_feat:
                clf_spatial = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
                clf_spatial.fit(support_features_spatial, support_ys)

                scores_spatial = clf_spatial.predict_proba(query_features_spatial)

            # train on the global features
            if opt.use_global_feat:
                clf_global = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs', max_iter=1000, multi_class='multinomial')
                clf_global.fit(support_features, support_ys)

                scores_global = clf_global.predict_proba(query_features)

            # aggregate the predictions
            if opt.use_spatial_feat and opt.use_global_feat and opt.aggregation == "max":
                query_ys_pred = np.stack([scores_global, scores_spatial], axis=1).max(1).argmax(1)
            else:
                query_ys_pred = (scores_global + scores_spatial).argmax(1)

            # compute the acc
            acc.append(metrics.accuracy_score(query_ys, query_ys_pred))

    return mean_confidence_interval(acc)





def meta_test_torch(model, testloader, opt):
    """
    meta testing loop using pytorch implementation of the classifier (might give slightly worse results)
    """
    model = model.eval()

    acc = []
    for idx, data in tqdm(enumerate(testloader), total=len(testloader)):
        # fetch the data
        support_xs, support_ys, query_xs, query_ys = data
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        support_ys = support_ys.cuda()

            # reshape the inputs
        bz, Ns, C, H, W = support_xs.shape
        bz, Nq, C, H, W = query_xs.shape
        assert bz == 1
        assert Ns == (opt.n_shots * opt.n_ways * opt.n_aug_support_samples)
        assert Nq == (opt.n_queries * opt.n_ways)
        support_xs = support_xs.view(-1, C, H, W).contiguous()
        query_xs = query_xs.view(-1, C, H, W).contiguous()
        query_ys = query_ys.view(-1).contiguous()

        support_ys_np = support_ys.view(-1).cpu().numpy()
        query_ys_np = query_ys.view(-1).cpu().numpy()

        # forward pass
        _, spatial_f_s, _, support_features = model(support_xs)
        _, spatial_f_q, _, query_features = model(query_xs)
        
        # initialize the linear classifier
        clf = BaselineFinetune(opt)
        
        scores_spatial, scores_global = 0.0, 0.0

        assert opt.use_spatial_feat or opt.use_global_feat, "choose the usage of either spa. of glob. features"

        # train on the spatial features
        if opt.use_spatial_feat:
            scores_spatial = clf(spatial_f_s.detach(), support_ys.detach(), spatial_f_q.detach(), spatial=True, weight_inprint=opt.weight_inprint)      

        # train on the global features
        if opt.use_global_feat:
            scores_global = clf(spatial_f_s.detach(), support_ys.detach(), spatial_f_q.detach(), spatial=False, weight_inprint=opt.weight_inprint)   

        # aggregate the prediction
        if opt.use_spatial_feat and opt.use_global_feat and opt.aggregation == "max":
            query_ys_pred = np.stack([scores_global, scores_spatial], axis=1).max(1).argmax(1)
        else:
            query_ys_pred = (scores_global + scores_spatial).argmax(1)

        # compute the acc for the current run
        acc.append(metrics.accuracy_score(query_ys_np, query_ys_pred))

    model = model.train()

    return mean_confidence_interval(acc)

class BaselineFinetune(nn.Module):
    """
    training the classifier for a single testing run
    """
    def __init__(self, opt):
        super(BaselineFinetune, self).__init__()
        self.n_way = opt.n_ways
        self.n_support = opt.n_aug_support_samples
        self.n_query = opt.n_queries
        self.feat_dim = 640
        self.alpha = 0.05 if self.n_way  == 1 else 0.01
        self.lr = 0.5 if self.n_way  == 1 else 1.0
        
    def forward(self, z_support, y_support, z_query, spatial=False, weight_inprint=False):
        torch.manual_seed(0)
        kernel_size = (z_support.size(2), z_support.size(3)) if spatial else None
        y_support = y_support.view(-1)

        if spatial:
            assert z_support.dim() == 4
            assert z_query.dim() == 4
        else:
            if z_support.dim() == 4:
                z_support = z_support.view(z_support.size(0), z_support.size(1), -1).mean(-1)
                z_query = z_query.view(z_query.size(0), z_query.size(1), -1).mean(-1)

        z_support = F.normalize(z_support, dim=1, p=2)
        z_query = F.normalize(z_query, dim=1, p=2)

        if spatial:
            linear_clf = nn.Sequential(nn.AdaptiveMaxPool2d(2), 
                                nn.Conv2d(self.feat_dim, self.n_way, kernel_size=(2,2), padding=0))
        else:
            linear_clf = nn.Linear(self.feat_dim, self.n_way)

        linear_clf = linear_clf.cuda()

        if weight_inprint:
            if spatial:
                z_support_pooled = nn.AdaptiveMaxPool2d(2)(z_support)
                prototypes = [z_support_pooled[y_support==l].mean(0) for l in y_support.unique()]
            else:
                prototypes = [z_support[y_support==l].mean(0) for l in y_support.unique()]
            prototypes = F.normalize(torch.stack(prototypes), dim=1, p=2)

            if spatial:
                linear_clf[1].weight.data.copy_(prototypes.data)
            else:
                linear_clf.weight.data.copy_(prototypes.data)

        set_optimizer =  torch.optim.LBFGS(linear_clf.parameters(), lr=self.lr)
        loss_function = nn.CrossEntropyLoss().cuda()
        
        def closure():
            set_optimizer.zero_grad()
            scores = linear_clf(z_support).squeeze()
            loss = loss_function(scores, y_support)

            l2_penalty = 0
            for param in linear_clf.parameters():
                l2_penalty = l2_penalty + 0.5 * (param ** 2).sum()
            loss = loss + self.alpha * l2_penalty

            loss.backward()
            
            return loss
    
        set_optimizer.step(closure)
        
        scores = linear_clf(z_query).squeeze().softmax(1).detach().cpu().numpy()
        return scores












