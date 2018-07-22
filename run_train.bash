python script/experiment/train.py \
-d '(6,)' \
-r 1 \
--dataset market1501 \
--ids_per_batch 8 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 151 \
--total_epochs 300 \
--exp_dir train_with_crop \
--set_seed True  \
--train_crop True \
--eval_crop True \
--train_down False
#--resume True \
#--trainset_part train
