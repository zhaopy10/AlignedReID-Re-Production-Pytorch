from .Dataset import Dataset
from ..utils.dataset_utils import parse_im_name

import os.path as osp
from PIL import Image
import numpy as np
from collections import defaultdict
import pickle

from ..utils.extend import extend_ims


class TrainSet(Dataset):
  """Training set for triplet loss.
  Args:
    ids2labels: a dict mapping ids to labels
  """

  def __init__(
      self,
      im_dir=None,
      im_names=None,
      ids2labels=None,
      ids_per_batch=None,
      ims_per_id=None,
      train_crop=False,
      train_down=False,
      train_padding=False,
      **kwargs):

    # The im dir of all images
    self.im_dir = im_dir
    self.im_names = im_names
    self.ids2labels = ids2labels
    self.ids_per_batch = ids_per_batch
    self.ims_per_id = ims_per_id

    im_ids = [parse_im_name(name, 'id') for name in im_names]
    #print('im_ids',im_ids)
    self.ids_to_im_inds = defaultdict(list)
    for ind, id in enumerate(im_ids):
      self.ids_to_im_inds[id].append(ind)
    #print('ids_to_im_inds',ids_to_im_inds)
    self.ids = self.ids_to_im_inds.keys()
    #print('ids',ids)
    
    self.train_crop = train_crop
    self.train_down = train_down
    self.train_padding = train_padding

    
    super(TrainSet, self).__init__(
      dataset_size=len(self.ids),
      batch_size=ids_per_batch,
      **kwargs)

  def get_sample(self, ptr):
    """Here one sample means several images (and labels etc) of one id.
    Returns:
      ims: a list of images
    """
    inds = self.ids_to_im_inds[self.ids[ptr]]
    if len(inds) < self.ims_per_id:
      inds = np.random.choice(inds, self.ims_per_id, replace=True)
    else:
      inds = np.random.choice(inds, self.ims_per_id, replace=False)
    im_names = [self.im_names[ind] for ind in inds]      
    ims = [np.asarray(Image.open(osp.join(self.im_dir, name)))
           for name in im_names]
    
    tmp_ims, extended_num = extend_ims(ims, crop=self.train_crop, down_sample=self.train_down, padding=self.train_padding)

    extended_ims, extended_mirrored = zip(*[self.pre_process_im(im) for im in tmp_ims])
    #ims, mirrored = zip(*[self.pre_process_im(im) for im in ims])
    labels = [self.ids2labels[self.ids[ptr]] for _ in range(self.ims_per_id)]
    
    extended_im_names = [name for name in im_names for i in range(extended_num)]
    extended_labels = [label for label in labels for i in range(extended_num)]
    #print('ims', type(ims), len(ims))
    # update code here
    #print('ims',ims)
    #print('im_names',im_names)
    #print('labels',labels)
    #print('mirrored',mirrored)
    
    return extended_ims, extended_im_names, extended_labels, extended_mirrored
    #return ims, im_names, labels, mirrored

  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done and self.shuffle:
      np.random.shuffle(self.ids)
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, im_names, labels, mirrored = zip(*samples)
    
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    ims = np.stack(np.concatenate(im_list))
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    im_names = np.concatenate(im_names)
    labels = np.concatenate(labels)
    mirrored = np.concatenate(mirrored)
    #print('ims', type(ims), ims.shape)
    return ims, im_names, labels, mirrored, self.epoch_done
