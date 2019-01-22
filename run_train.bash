python script/experiment/train.py \
-d '(3,)' \
-r 1 \
--dataset owlii \
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
--exp_dir owlii_1017_with_mobilenetv2_0.75_pyz_temp \
--set_seed False  \
--log_steps \

#--resume True \
#--trainset_part train
#--ids_per_batch 32 \
