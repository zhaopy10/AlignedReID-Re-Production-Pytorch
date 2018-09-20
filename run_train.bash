python script/experiment/train.py \
-d '(0,)' \
-r 1 \
--dataset market1501 \
--ids_per_batch 16 \
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
--exp_dir market_with_mobilenetv2_relu_075 \
--set_seed True  \
--train_crop False \
--eval_crop False \
--train_down False \
--eval_down False
#--resume True \
#--trainset_part train
#--ids_per_batch 32 \
