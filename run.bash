python script/experiment/train.py \
-d '(1,)' \
--dataset owlii \
--normalize_feature true \
-glw 1 \
-llw 0 \
-idlw 0 \
--only_test true \
--set_seed true \
--model_weight_file ./owlii_1017_with_mobilenetv2_1.0_pyz/model_weights.pth \
--exp_dir eval_test \
--eval_crop false \
--eval_down false

#--model_weight_file ~/reid/pretrained_model/model_weight.pth  \
