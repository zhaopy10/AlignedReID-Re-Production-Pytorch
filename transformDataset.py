
from __future__ import print_function

import sys
sys.path.insert(0, '/home/corp.owlii.com/peiyao.zhao/reid/AlignedReID-Re-Production-Pytorch')

from zipfile import ZipFile
import os.path as osp
import numpy as np

from aligned_reid.utils.utils import may_make_dir
from aligned_reid.utils.utils import save_pickle
from aligned_reid.utils.utils import load_pickle

from aligned_reid.utils.dataset_utils import get_im_names
from aligned_reid.utils.dataset_utils import partition_train_val_set
from aligned_reid.utils.dataset_utils import new_im_name_tmpl
from aligned_reid.utils.dataset_utils import parse_im_name as parse_new_im_name
from aligned_reid.utils.dataset_utils import move_ims

from shutil import copyfile

def load_image_list(filename):
    line_list = []
    with open(filename) as f:
        line = f.readline()
        line_list.append(line)
        while line:
            line = f.readline()
            if len(line) > 0:
                line_list.append(line)
    name_list = []
    id_list = []
    for i in range(0, len(line_list)):
        splited_str = line_list[i].split()
        name_list.append(splited_str[0])
        id_list.append(int(splited_str[1]))
    return name_list, id_list

def convert_name(ori_name, id_offset=0):
    id = int(ori_name[0:4])
    index = int(ori_name[10:13])
    cam = int(ori_name[14:16])
    new_name = '%08d_%04d_%08d.jpg'%(id+id_offset, cam, index)
    #print('id',id,'index',index,'cam',cam)
    return new_name

def handle_test_set(copy=False):
    test_im_names = []
    test_ids = []
    
    query_im_names, query_ids = load_image_list('list_query.txt')
    test_im_names.extend(query_im_names)
    test_ids.extend(query_ids)
    gallery_im_names, gallery_ids = load_image_list('list_gallery.txt')
    test_im_names.extend(gallery_im_names)
    test_ids.extend(gallery_ids)
    test_marks = [0 for _ in range(len(query_ids) + len(gallery_ids))]
    test_marks[len(query_ids):] = [1 for _ in range(len(gallery_ids))]
    
    test_im_names, test_marks = zip(*sorted(zip(test_im_names, test_marks)))
    test_im_names_converted = []
    for name in test_im_names:
        new_name = convert_name(name)
        test_im_names_converted.append(new_name)
        if copy:
            copyfile('test/'+name, 'images/' + new_name)
    max_id = max(test_ids) + 1
    return test_im_names_converted, test_marks, max_id

def handle_trainval_set(id_offset=0, copy=False):
    trainval_im_names = []
    trainval_ids = []
    trainval_lables = []
    train_im_names, train_ids = load_image_list('list_train.txt')
    trainval_im_names.extend(train_im_names)
    trainval_ids.extend([id + id_offset for id in train_ids])
    trainval_lables.extend(train_ids)
    val_im_names, val_ids = load_image_list('list_val.txt')
    trainval_im_names.extend(val_im_names)
    trainval_ids.extend([id + id_offset for id in val_ids])
    trainval_lables.extend(val_ids)
    
    trainval_im_names, trainval_ids, trainval_lables = zip(*sorted(zip(trainval_im_names, trainval_ids, trainval_lables)))
    trainval_im_names_converted = []
    for name in trainval_im_names:
        new_name = convert_name(name, id_offset)
        trainval_im_names_converted.append(new_name)
        if copy:
            copyfile('train/'+name, 'images/' + new_name)
            
    trainval_ids2labels = dict(zip(trainval_ids, trainval_lables))
    
    return trainval_im_names_converted, trainval_ids2labels

if __name__ == '__main__':
    test_im_names, test_marks, id_offset = handle_test_set(copy=False)
    trainval_im_names, trainval_ids2labels = handle_trainval_set(id_offset=id_offset, copy=False)
    #print(test_im_names[0:100])
    #print(test_marks[0:100])
    print('test_size', len(test_im_names))
    print('trainval_size', len(trainval_im_names))
    
    partitions = {'trainval_im_names':trainval_im_names,
                 'trainval_ids2labels':trainval_ids2labels,
                 'test_im_names':test_im_names,
                 'test_marks':test_marks}
    save_pickle(partitions, 'partitions.pkl')
    
    
    
