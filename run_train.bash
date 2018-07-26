python script/experiment/train.py \
-d '(5,6,)' \
-r 1 \
--dataset combined \
--ids_per_batch 32 \
--ims_per_id 4 \
--normalize_feature false \
-gm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 200 \
--total_epochs 400 \
--exp_dir train_with_combined \
--set_seed True  \
--train_crop False \
--eval_crop False \
--train_down False
#--resume True \
#--trainset_part train
