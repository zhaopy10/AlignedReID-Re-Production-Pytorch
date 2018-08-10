python script/experiment/train.py \
-d '(4,)' \
--dataset market1501 \
--normalize_feature false \
-glw 1 \
-llw 0 \
-idlw 0 \
--only_test true \
--set_seed true \
--model_weight_file ./train_with_combined/model_weights.pth \
--exp_dir eval_test \
--eval_crop false \
--eval_down false

#--model_weight_file ~/reid/pretrained_model/model_weight.pth  \
