from torch.utils.data import Dataset
from PIL import Image
import os
import h5py
import numpy as np
import json

class nyuv(Dataset):
    def __init__(self, split="train", num_im = -1, transforms=None):
        if split == 'train':
            self.data_dir = '../../data/NYUv2/image/train'
        elif split == 'test':
            self.data_dir = '../../data/NYUv2/image/test'
        else:
            raise ValueError('Invalid split')    
        self.transforms = transforms
        self.image_file = os.path.join(self.data_dir, "lmdb_nyuv_"+split+".h5")
        self.im_h5      = h5py.File(self.image_file, 'r')
        self.im_refs    = self.im_h5['images']
        self.names      = self.im_h5['filenames']
        if num_im > -1:
            self.image_index = self.im_h5['valid_idx'][:num_im]
        else:
            self.image_index = self.im_h5['valid_idx']
        self.im_sizes    = np.vstack((self.im_h5['image_widths'],self.im_h5['image_heights']))
        self.info        = json.load(open(os.path.join('../../data/visual-genome', "VG-SGG-dicts.json"), 'r'))
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind   = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:self.class_to_ind[k])

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
    def _im_getter(self, idx):
        w, h = self.im_sizes[:,idx]
        ridx = self.image_index[idx]
        im   = self.im_refs[ridx]
        im   = im[:, :h, :w] # crop out
        im   = im.transpose((1,2,0)) # c h w -> h w c
        return im
        
    def __len__(self):
        return len(self.image_index)
    
    def __getitem__(self, index):
        img     = Image.fromarray(self._im_getter(index)); width, height = img.size
        img, _ = self.transforms(img, img)
        
        return img, None, self.names[index]
    
    def get_img_info(self, img_id):
        w, h = self.im_sizes[:, img_id]
        return {"height": h, "width": w}
    
if __name__ == '__main__':
    dataset = nyuv(split='test')
    a = dataset[1]
    print('ok')