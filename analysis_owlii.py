import os
import pickle
import numpy as np

f = open('eval_owlii/owlii_1.0_all.pkl')
data = pickle.load(f)
f.close()

gt_label = data['gt_label']
result = data['result']
dist_vec = data['dist_vec']

same_label_dist = [x for i,x in enumerate(dist_vec) if gt_label[i]==1]
diff_label_dist = [x for i,x in enumerate(dist_vec) if gt_label[i]==0]
print('same_label_dist', np.mean(same_label_dist))
print('diff_label_dist', np.mean(diff_label_dist))


same_label_num = gt_label.count(1)
diff_label_num = gt_label.count(0)
print('same_label_num', same_label_num)
print('diff_label_num', diff_label_num)

re_keys = result.keys()
re_keys.sort()
for key in re_keys:
    detected_label = result[key]
    print('threshold', key)
    same_to_diff_list = [1 for i in range(len(gt_label)) if gt_label[i]==1 and detected_label[i]==0]
    same_to_same_list = [1 for i in range(len(gt_label)) if gt_label[i]==1 and detected_label[i]==1]
    print('same condition', len(same_to_diff_list), len(same_to_same_list))
    diff_to_same_list = [1 for i in range(len(gt_label)) if gt_label[i]==0 and detected_label[i]==1]
    diff_to_diff_list = [1 for i in range(len(gt_label)) if gt_label[i]==0 and detected_label[i]==0]
    print('diff condition', len(diff_to_same_list), len(diff_to_diff_list))
    print('same condition error rate', len(same_to_diff_list) / float(same_label_num))
    print('diff condition error rate', len(diff_to_same_list) / float(diff_label_num))
