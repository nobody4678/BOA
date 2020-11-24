import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join
import glob
import joblib
import constants


class pw3d_hmmr_dataset(Dataset):
    def __init__(self, datapath=None):
        """
        ['N', 'centers', 'kps', 'gt3ds', 'images', 'im_shapes', 'im_paths', 'poses', 'scales', 'shape', 'start_pts', 'time_pts']
        """
        super(pw3d_hmmr_dataset, self).__init__()
        if datapath is None:
            self.data = joblib.load('data/dataset_extras/3dpw_test_all.pt')
        else:
            self.data = joblib.load(datapath)
        self.data_num = self.data['N']
        self.images = self.data['images']
        self.js2d = self.data['kps']
        self.js3d = self.data['gt3ds']
        self.betas = self.data['shape']
        self.thetas = self.data['poses'] # N, 24, 3
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        print('3dpw hmmr version length:', self.data_num)
    
    def process_j2d(self, kp):
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        kp = kp.astype('float32')
        return kp
    
    def __getitem__(self, index):
        item = {}
        rgb_img = self.images[index].copy()
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        rgb_img = torch.from_numpy(rgb_img).float()
        ori_rgb_img = self.images[index].copy()
        ori_rgb_img = np.transpose(ori_rgb_img.astype('float32'),(2,0,1))/255.0
        ori_rgb_img = torch.from_numpy(ori_rgb_img).float()
        item['pose'] = torch.from_numpy(self.thetas[index].copy()).float()
        item['betas'] = torch.from_numpy(self.betas[index].copy()).float()
        item['img'] = self.normalize_img(rgb_img)
        item['ori_img'] = self.normalize_img(ori_rgb_img)

        item['pose_3d'] = torch.from_numpy(self.js3d[index].copy()).float()
        item['keypoints'] = torch.from_numpy(self.process_j2d(self.js2d[index].copy())).float()
        item['gender'] = ''
        return item
        
    def __len__(self):
        return self.data_num

