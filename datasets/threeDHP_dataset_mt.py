"""
Only for mean teacher
"""

from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np
import cv2
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from utils.image_aug_utils import augment_appearance
from utils.image_background_aug_utils import augment_background
import random
import time

class HP3D(Dataset):
    """
    'imgname', 'center', 'scale', 'part', 'S'
    """
    def __init__(self, options, dataset, num_aug, occlude_aug=False):
        super(HP3D, self).__init__()
        self.dataset = dataset
        self.options = options
        self.occlude_aug = occlude_aug
        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[0][dataset])
        self.num_aug = num_aug

        # load attributes
        self.imgname = self.data['imgname']
        self.scale = self.data['scale']
        self.center = self.data['center']
        # self.pose = self.data['pose'].astype(np.float)
        # self.betas = self.data['shape'].astype(np.float)
        self.pose_3d = self.data['S']
        # self.smpl_j2ds = self.data['smpl_j2d']

        keypoints_gt = self.data['part']
        keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # gender = self.data['gender']
        # self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        self.length = self.scale.shape[0]

    def augm_params(self, istrain):
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if istrain:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
        return flip, pn, rot, sc

    def read_image(self, imgname):
        img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        return img

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, is_train):
        rgb_img = crop(rgb_img.copy(), center, scale, [constants.IMG_RES, constants.IMG_RES], rot=rot)
        if is_train:
            if flip:
                rgb_img = flip_img(rgb_img)
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img

    def j2d_processing(self, kp, center, scale, r, f, is_train):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if is_train and f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f, is_train):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if is_train and f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f, is_train):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        if is_train:
            # rotation or the pose parameters
            pose[:3] = rot_aa(pose[:3], r)
            # flip the pose parameters
            if f:
                pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def process_sample(self, image, keypoints, S, center, scale, flip, pn, rot, sc, is_train):
        # labeled keypoints
        kp2d = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip, is_train=is_train)).float()
        # smpl_j2ds = torch.from_numpy(self.j2d_processing(smpl_j2ds, center, sc*scale, rot, flip, is_train=is_train)).float()
        img = self.rgb_processing(image, center, sc*scale, rot, flip, pn, is_train=is_train)
        img = torch.from_numpy(img).float()
        img = self.normalize_img(img)
        # pose = torch.from_numpy(self.pose_processing(pose, rot, flip, is_train=is_train)).float()
        # betas = torch.from_numpy(beta).float()
        S = torch.from_numpy(self.j3d_processing(S, rot, flip, is_train=is_train)).float()
        return kp2d, img, S

    def __getitem__(self, index):
        item = {}

        kp2d_all, img_all, pose_all, betas_all, pose_3d_all, smpl_kp2d_all = [], [], [], [], [], []

        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        # pose = self.pose[index].copy()
        # betas = self.betas[index].copy()
        imgname = join(self.img_dir, self.imgname[index])
        img = self.read_image(imgname).copy()
        S = self.pose_3d[index].copy()
        # smpl_j2ds = self.smpl_j2ds[index].copy()

        # ori image, no aug
        flip, pn, rot, sc = 0, np.ones(3), 0, 1
        kp2d_i, img_i, S_i = self.process_sample(img.copy(), keypoints, S, center, scale, flip, pn, rot, sc, is_train=False)

        item['oriimg'] = img_i
        # item['oripose'] = pose_i
        # item['oribeta'] = betas_i
        item['orikeypoints'] = kp2d_i
        item['oripose_3d'] = S_i
        # item['orismpl_j2ds'] = smpl_j2ds_i
        kp2d_all.append(kp2d_i)
        img_all.append(img_i)
        # pose_all.append(pose_i)
        # betas_all.append(betas_i)
        pose_3d_all.append(S_i)
        # smpl_kp2d_all.append(smpl_j2ds_i)

        for _ in range(self.num_aug):
            flip, pn, rot, sc = self.augm_params(istrain=True)
            scale = self.scale[index].copy()
            center = self.center[index].copy()
            keypoints = self.keypoints[index].copy()
            # pose = self.pose[index].copy()
            # betas = self.betas[index].copy()
            smpl_j2ds = self.smpl_j2ds[index].copy()

            kp2d_i, img_i, S_i = self.process_sample(img.copy(), keypoints, S, center, scale, flip, pn, rot, sc, is_train=True)

            kp2d_all.append(kp2d_i)
            img_all.append(img_i)
            # pose_all.append(pose_i)
            # betas_all.append(betas_i)
            pose_3d_all.append(S_i)
            # smpl_kp2d_all.append(smpl_j2ds_i)
        
        item['keypoints'] = torch.stack(kp2d_all)
        item['img'] = torch.stack(img_all)
        # item['pose'] = torch.stack(pose_all)
        # item['betas'] = torch.stack(betas_all)
        item['pose_3d'] = torch.stack(pose_3d_all)
        # item['smpl_j2ds'] = torch.stack(smpl_kp2d_all)
        # item['gender'] = self.gender[index]
        return item

    def __len__(self):
        return len(self.imgname)
