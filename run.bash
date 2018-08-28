python script/experiment/train.py \
-d '(4,)' \
--dataset douyin \
--normalize_feature false \
-glw 1 \
-llw 0 \
-idlw 0 \
--only_test true \
--set_seed true \
--model_weight_file ./douyin_0821/model_weights.pth \
--exp_dir eval_test \
--eval_crop True \
--eval_down false

#--model_weight_file ~/reid/pretrained_model/model_weight.pth  \
