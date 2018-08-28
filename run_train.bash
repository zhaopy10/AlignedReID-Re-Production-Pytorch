python script/experiment/train.py \
-d '(1,2,3,6,)' \
-r 1 \
--dataset douyin \
--ids_per_batch 32 \
--ims_per_id 8 \
--normalize_feature false \
-gm 0.3 \
-glw 1 \
-llw 0 \
-idlw 0 \
--base_lr 2e-4 \
--lr_decay_type exp \
--exp_decay_at_epoch 150 \
--total_epochs 300 \
--exp_dir douyin_0821_extend_only_up_all \
--set_seed True  \
--train_crop True \
--eval_crop True \
--train_down False \
--eval_down False
#--resume True \
#--trainset_part train
