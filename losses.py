
import torch
import torch.nn as nn
import torch.nn.functional as F

def contrast_distill(f1, f2):
    """
    Contrastive Distillation
    """
    f1 = F.normalize(f1, dim=1, p=2)
    f2 = F.normalize(f2, dim=1, p=2)
    loss = 2 - 2 * (f1 * f2).sum(dim=-1)
    return loss.mean()


class DistillKL(nn.Module):
    """
    KL divergence for distillation
    """
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss (based on https://github.com/HobbitLong/SupContrast)
    """
    def __init__(self, temperature=None):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def _compute_logits(self, features_a, features_b, attention=None):
        # global similarity
        if features_a.dim() == 2:
            features_a = F.normalize(features_a, dim=1, p=2)
            features_b = F.normalize(features_b, dim=1, p=2)
            contrast = torch.matmul(features_a, features_b.T)

        # spatial similarity
        elif features_a.dim() == 4:
            contrast = attention(features_a, features_b)

        else:
            raise ValueError
        
        # note here we use inverse temp
        contrast = contrast * self.temperature 
        return contrast

    def forward(self, features_a, features_b=None, labels=None, attention=None):
        device = (torch.device('cuda') if features_a.is_cuda else torch.device('cpu'))
        num_features, num_labels = features_a.shape[0], labels.shape[0]

        # using only the current features in a given batch
        if features_b is None:
            features_b = features_a
            # mask to remove self contrasting
            logits_mask = (1. - torch.eye(num_features)).to(device)
        else:
            # contrasting different features (a & b), no need to mask the diagonal
            logits_mask = torch.ones(num_features, num_features).to(device)
        
        # mask to only maintain positives
        if labels is None:
            # standard self supervised case
            mask = torch.eye(num_labels, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        # replicate the mask since the labels are just for N examples
        if num_features != num_labels:
            assert num_labels * 2 == num_features
            mask = mask.repeat(2, 2)

        # compute logits
        contrast = self._compute_logits(features_a, features_b, attention)

        # remove self contrasting
        mask = mask * logits_mask

        # normalization over number of positives
        normalization = mask.sum(1)
        normalization[normalization == 0] = 1.

        # for stability
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        exp_logits = exp_logits * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / normalization
        loss = -mean_log_prob_pos.mean()

        return loss

