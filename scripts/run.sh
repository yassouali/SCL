# ======================
#     miniImageNet
# ======================

# contrastive pre-training
python train_contrastive.py --data_root DATA_PATH --dataset miniImageNet --model_name train_miniImageNet \
                        --epochs 90 --learning_rate 5e-2 --batch_size 64 --lr_decay_epochs 60,80 \
                        --lambda_cls 1.0 --lambda_spatial 1.0 --spatial_cont_loss

# distillation
python train_distillation.py --data_root DATA_PATH  --dataset miniImageNet --model_name distill_miniImageNet \
                        --epochs 90 --learning_rate 1e-2 --batch_size 64 \
                        --lr_decay_epochs 60,80 --lambda_contrast_g 10.0 --lambda_KD 1.0 \
                        --model_path_t models_pretrained/train_miniImageNet/resnet12_last.pth

# evaluation (replace MODEL_PATH with either the pretrained model or the distilled model)
python eval_fewshot.py --use_spatial_feat --use_global_feat --aggregation sum \
                        --model_path MODEL_PATH --data_root DATA_PATH \
                        --dataset miniImageNet --n_shots 1 --n_test_runs 600







# ======================
#     tieredImageNet
# ======================

# contrastive pre-training
python train_contrastive.py --data_root DATA_PATH  --dataset tieredImageNet --model_name train_tieredImageNet \
                        --epochs 60 --learning_rate 5e-2 --batch_size 64 --lr_decay_epochs 30,40,50 \
                        --lambda_cls 1.0 --lambda_spatial 1.0 --spatial_cont_loss

# distillation
python train_distillation.py --data_root DATA_PATH --dataset tieredImageNet --model_name distill_tieredImageNet \
                        --epochs 60 --learning_rate 1e-2 --batch_size 64 \
                        --lr_decay_epochs 30,40,50 --lambda_contrast_g 10.0 --lambda_KD 1.0 \
                        --model_path_t models_pretrained/train_tieredImageNet/resnet12_last.pth

# evaluation (replace MODEL_PATH with either the pretrained model or the distilled model)
python eval_fewshot.py --use_spatial_feat --use_global_feat --aggregation sum \
                        --model_path MODEL_PATH --data_root DATA_PATH \
                        --dataset tieredImageNet --n_shots 1 --n_test_runs 600







# ======================
#         FC100
# ======================

# contrastive pre-training
python train_contrastive.py --data_root DATA_PATH --dataset FC100 --model_name train_FC100 \
                        --epochs 90 --learning_rate 5e-2 --batch_size 64 \
                        --lr_decay_epochs 45,60,75 --lambda_cls 1.0 --lambda_spatial 1.0 --spatial_cont_loss

# distillation
python train_distillation.py --data_root DATA_PATH --dataset FC100 --model_name distill_FC100 \
                        --epochs 90 --learning_rate 1e-2 --batch_size 64 \
                        --lr_decay_epochs 45,60,75 --lambda_contrast_g 10.0 --lambda_KD 1.0 \
                        --model_path_t models_pretrained/train_FC100/resnet12_last.pth

# evaluation (replace MODEL_PATH with either the pretrained model or the distilled model)
python eval_fewshot.py --use_spatial_feat --use_global_feat --aggregation sum \
                        --model_path MODEL_PATH --data_root DATA_PATH \
                        --dataset FC100 --n_shots 1 --n_test_runs 600 







# ======================
#      CIFAR-FS
# ======================

# contrastive pre-training
python train_contrastive.py --data_root DATA_PATH --dataset CIFAR-FS --model_name train_CIFARFS \
                        --epochs 90 --learning_rate 5e-2 --batch_size 64 \
                        --lr_decay_epochs 45,60,75 --lambda_cls 0.5 --lambda_spatial 0.5 --spatial_cont_loss


# distillation
python train_distillation.py -data_root DATA_PATH --dataset CIFAR-FS --model_name distill_CIFARFS \
                        --epochs 90 --learning_rate 1e-2 --batch_size 64 \
                        --lr_decay_epochs 45,60,75 --lambda_contrast_g 10.0 --lambda_KD 1.0 \
                        --model_path_t models_pretrained/train_CIFARFS/resnet12_last.pth

# evaluation (replace MODEL_PATH with either the pretrained model or the distilled model)
python eval_fewshot.py --use_spatial_feat --use_global_feat --aggregation sum \
                        --model_path MODEL_PATH --data_root DATA_PATH \
                        --dataset CIFAR-FS --n_shots 1 --n_test_runs 600







# ======================
#      Cross-domain
# ======================

# contrastive pre-training
python train_contrastive.py --data_root DATA_PATH --dataset cross --model_name train_crossdomain \
                        --epochs 90 --learning_rate 5e-2 --batch_size 64 \
                        --lr_decay_epochs 60,80 --lambda_cls 1.0 --lambda_spatial 1.0 --spatial_cont_loss

# distillation
python train_distillation.py --data_root DATA_PATH --dataset cross --model_name distill_crossdomain \
                        --epochs 90 --learning_rate 1e-2 --batch_size 64 \
                        --lr_decay_epochs 45,60,75 --lambda_contrast_g 10.0 --lambda_KD 1.0 \
                        --model_path_t models_pretrained/train_crossdomain/resnet12_last.pth

# evaluation (replace MODEL_PATH with either the pretrained model or the distilled model)
# also set DATASET_NAME to one of the 4 possible choices: cub, cars, places and plantae 
python eval_fewshot.py --cross --use_spatial_feat --use_global_feat --aggregation sum \
                        --model_path MODEL_PATH --data_root DATA_PATH \
                        --dataset DATASET_NAME --n_shots 1 --n_test_runs 600 \
                        --num_workers 6 --n_aug_support_samples 10
