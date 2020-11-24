import sys
sys.path.append('.')

import os
import cv2
import copy
import time
import torch
import random
import joblib
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
from tqdm import tqdm
import learn2learn as l2l
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import config
import constants
from models import hmr, SMPL
from datasets import H36M, PW3D, HP3D
from utils.pose_utils import reconstruction_error
from utils.geometry import perspective_projection, rotation_matrix_to_angle_axis, batch_rodrigues
from smplify.prior import MaxMixturePrior

parser = argparse.ArgumentParser()
parser.add_argument('--expdir', type=str, default='newmaml-exps-gsy', help='common dir of each experiment')
parser.add_argument('--name', type=str, default='', help='exp name')
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--model_file', type=str, default='logs/GN-adv-lsgan-0root-v2-loss5-2stages-v2-fintune/checkpoints/2020_10_29-12_29_41.pt', help='base model')
parser.add_argument('--num_augsamples', type=int, default=0, help='times of augmentation')
parser.add_argument('--batch_size', type=int, default=1, help='')
parser.add_argument('--dataset_name', type=str, default='3dpw', choices=['3dpw', 'mpi-inf-3dhp'], help='test set name')
parser.add_argument('--img_res', type=int, default=224, help='image resolution')
parser.add_argument('--T', type=int, default=1, help='times of adaptation')
parser.add_argument('--offline', action='store_true', default=False, help='offline adapt?')

## baseline hyper-parameters
parser.add_argument('--lr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='adam beta2')
parser.add_argument('--use_mixtrain', action='store_true', default=False)
parser.add_argument('--s2dsloss_weight', type=float, default=10, help='weight of reprojection kp2d loss')
parser.add_argument('--shapepriorloss_weight', type=float, default=1e-5, help='weight of shape prior')
parser.add_argument('--gmmpriorloss_weight', type=float, default=2e-4, help='weight of pose prior(GMM)')
parser.add_argument('--labelloss_weight', type=float, default=1, help='weight of h36m loss')

## mean-teacher hyper-parameters
parser.add_argument('--use_meanteacher', action='store_true', default=False)
parser.add_argument('--ema_decay', type=float, default=0.3, help='ema_decay * T + (1-ema_decay) * M')
# fixed
parser.add_argument('--consistentloss_weight', type=float, default=0.01, help='weight of consistent loss')
parser.add_argument('--consistent_s3d_weight', type=float, default=5, help='weight of shape prior')
parser.add_argument('--consistent_s2d_weight', type=float, default=5, help='weight of consistent loss')
parser.add_argument('--consistent_pose_weight', type=float, default=1, help='weight of pose prior(GMM)')
parser.add_argument('--consistent_beta_weight', type=float, default=0.001, help='weight of h36m loss')

## bilevel hyper parameters
parser.add_argument('--use_bilevel', action='store_true', default=False)
parser.add_argument('--use_motionloss', action='store_true', default=False)
parser.add_argument('--metalr', type=float, default=3e-6, help='learning rate')
parser.add_argument('--prev_n', type=int, default=5)
parser.add_argument('--motionloss_weight', type=float, default=0.1)
parser.add_argument('--only_use_motionloss', action='store_true', default=False)


# predefined variables
device = torch.device('cuda')
J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
smpl_neutral = SMPL(config.SMPL_MODEL_DIR, create_transl=False).to(device)
smpl_male = SMPL(config.SMPL_MODEL_DIR, gender='male', create_transl=False).to(device)
smpl_female = SMPL(config.SMPL_MODEL_DIR, gender='female', create_transl=False).to(device)
# -- end

# tools of mean teacher 
def create_model(ema=False):
    model = hmr(config.SMPL_MEAN_PARAMS)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
# -- end

# other tools
def seed_everything(self, seed=42): # 42
    """ we need set seed to ensure that all model has same initialization
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    print('seed has been set')

# -- end


class Adaptor():
    def __init__(self, options):
        # prepare
        self.options = options
        self.exppath = osp.join(self.options.expdir, self.options.name)
        self.device = torch.device('cuda')
        # set seed
        seed_everything(self.options.seed)

        # build model and optimizer
        model = create_model()
        # using the tool of learn2learn to realize bilevel optimization
        if self.options.use_bilevel:
            self.model = l2l.algorithms.MAML(model, lr=self.options.metalr, first_order=False).to(self.device)
        else:
            self.model = model.to(self.device)
        # create a teacher model, whose initial weight is the copy of base model
        if self.options.use_meanteacher:
            ema_model = create_model(ema=True) # teacher model
            self.ema_model = ema_model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.options.lr, betas=(self.options.beta1, self.options.beta2))
        print('model & optimizer are set.')

        # load pretrained model (base model)
        checkpoint = torch.load(self.options.model_file)
        self.modeldict_copy = checkpoint['model']
        checkpoint['model'] = {k.replace('module.',''):v for k,v in checkpoint['model'].items()}
        self.model.load_state_dict(checkpoint['model'], strict=True)
        if self.options.use_meanteacher:
            checkpoint['model'] = {k.replace('module.',''):v for k,v in checkpoint['model'].items()}
            self.ema_model.load_state_dict(checkpoint['model'], strict=True)
        print('pretrained CKPT has been load')

        # build dataloders
        if '3dpw' in self.options.dataset_name:
            # 3dpw
            self.pw3d_dataset = PW3D(self.options, '3dpw', num_aug=self.options.num_augsamples)
            self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=1, shuffle=False, num_workers=8)
        elif 'mpi-inf' in self.options.dataset_name:
            # 3DHP
            self.pw3d_dataset = HP3D(self.options, 'mpi-inf-3dhp', num_aug=self.options.num_augsamples)
            self.pw3d_dataloader = DataLoader(self.pw3d_dataset, batch_size=1, shuffle=False, num_workers=8)
        # h36m
        self.h36m_dataset = H36M(self.options, 'h36m', num_aug=self.options.num_augsamples)
        self.h36m_dataloader = DataLoader(self.h36m_dataset, batch_size=1, shuffle=False, num_workers=8) #self.options.batch_size, shuffle=False, num_workers=8)
        print('dataset has been created')

        # prepare criterion functions
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_consistent = nn.MSELoss().to(self.device)
        self.criterion_poseprior = MaxMixturePrior(prior_folder='data',
                                                   num_gaussians=8,
                                                   dtype=torch.float32).to(self.device)
        print('loss funtion has been created')

    ### helper functions
    def decode_smpl_params(self, rotmats, betas, cam, neutral=True, pose2rot=False):
        if neutral:
            smpl_out = smpl_neutral(betas=betas, body_pose=rotmats[:,1:], global_orient=rotmats[:,0].unsqueeze(1), pose2rot=pose2rot)
        return {'s3d': smpl_out.joints, 'vts': smpl_out.vertices}

    def set_dropout_eval(self, m):
        classname = m.__class__.__name__
        if classname.find('Dropout') != -1:
            # print('freezing: {}'.format(classname))
            m.eval()

    def freeze_dropout(self,):
        self.model.apply(self.set_dropout_eval)
        if self.options.use_meanteacher:
            self.ema_model.apply(self.set_dropout_eval)

    ### helper functions end
    

    def inference(self):
        joint_mapper_h36m = constants.H36M_TO_J17 if self.options.dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if self.options.dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14

        # build human 3.6m loader if using the source data during online adaptation. 
        if self.options.use_mixtrain:
            h36m_loader = iter(self.h36m_dataloader)

        # if use the motion loss, we create a dict to save the previous images and its 2D keypoints.
        if self.options.use_motionloss:
            self.history_info = {}

        mpjpe, pampjpe, pck = [], [], []
        self.global_step = 0
        h36m_batch = None

        for step, pw3d_batch in tqdm(enumerate(self.pw3d_dataloader), total=len(self.pw3d_dataloader)):
            # for each arrived frames, we first adapt the history model, and then use the adapted model to estimate the human mesh.

            self.global_step = step

            # move test data to the gpu device
            pw3d_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in pw3d_batch.items()}

            # load source data, and move them to the gpu device
            if self.options.use_mixtrain:
                # load h36m data
                try:
                    h36m_batch = next(h36m_loader)
                except StopIteration:
                    h36m_loader = iter(self.h36m_dataloader)
                    h36m_batch = next(h36m_loader)
                h36m_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in h36m_batch.items()}

            # set model to the training mode
            self.model.train()
            if self.options.use_meanteacher:
                self.ema_model.train()
            # during adaptation, we don't use dropout
            self.freeze_dropout()

            # Step1. begin online adaptation
            # T = 1 in our experiments. 
            T = self.options.T
            for i in range(T):
                self.optimizer.zero_grad()
                adaptation_loss = self.meta_adapt(pw3d_batch, h36m_batch)
                adaptation_loss.backward()
                self.optimizer.step()
            # exponential moving averge update. (teacher model)
            if self.options.use_meanteacher:
                update_ema_variables(self.model, self.ema_model, self.options.ema_decay, self.global_step)

            # Step2. begin test
            eval_res = self.test(pw3d_batch, joint_mapper_gt, joint_mapper_h36m)
            mpjpe.append(eval_res['mpjpe'])
            pampjpe.append(eval_res['pa-mpjpe'])
            pck.append(eval_res['pck'])
        
        print('=== Final Results ===')
        print('MPJPE:', np.mean(mpjpe)*1000)
        print('PAMPJPE:', np.mean(pampjpe)*1000)
        print('PCK:', pck.mean()*100)
        mpjpe = np.stack(mpjpe)
        pampjpe = np.stack(pampjpe)
        pck = np.stack(pck)
        np.save(osp.join(self.exppath, 'mpjpe'), mpjpe)
        np.save(osp.join(self.exppath, 'pampjpe'), pampjpe)
        np.save(osp.join(self.exppath, 'pck'), pck)

    def meta_adapt(self, unlabeled_batch, labeled_batch=None):

        # lower-level weight probe
        if self.options.use_bilevel:
            learner = self.model.clone()
        total_loss = self.adaptation(learner, unlabeled_batch, labeled_batch, use_motionloss=False, use_consistentLoss=False)
        if self.options.use_bilevel:
            learner.adapt(total_loss)
        

        # upper-level model update
        total_loss = self.adaptation(learner, unlabeled_batch, labeled_batch, use_motionloss=self.options.use_motionloss, use_consistentLoss=True, only_use_motionloss=self.options.only_use_motionloss)
        return total_loss
    
    def adaptation(self, learner, unlabeled_batch, labeled_batch=None, use_motionloss=False, use_consistentLoss=False, only_use_motionloss=False):
        
        # adapt unlabeled data, short for udata
        if self.options.dataset_name == '3dpw':
            uimage, us2d = unlabeled_batch['img'].squeeze(0), unlabeled_batch['smpl_j2ds'].squeeze(0)
        elif self.options.dataset_name == 'mpi-inf-3dhp':
            uimage, us2d = unlabeled_batch['img'].squeeze(0), unlabeled_batch['keypoints'].squeeze(0)
            
        if use_motionloss:
            # if consider motion loss, we need store history data.
            history_idx = self.global_step - self.options.prev_n
            if history_idx > 0:
                hist_uimage, hist_us2d = self.history_info[history_idx]['image'].to(self.device),\
                                         self.history_info[history_idx]['s2d'].to(self.device)
            else:
                hist_uimage, hist_us2d = None, None
            unlabelloss = self.adapt_for_unlabeled_data(learner, uimage, us2d, hist_uimage, hist_us2d,use_consistentLoss=use_consistentLoss,only_use_motionloss=only_use_motionloss)
            self.history_info[self.global_step] = {'image': uimage.clone().detach().cpu(), 's2d': us2d.clone().detach().cpu()}
            if labeled_batch is not None:
                # update for labeled data
                h36image, h36s3d, h36s2d, h36beta, h36pose = labeled_batch['img'].squeeze(0),\
                                                            labeled_batch['pose_3d'].squeeze(0),\
                                                            labeled_batch['keypoints'].squeeze(0),\
                                                            labeled_batch['betas'].squeeze(0),\
                                                            labeled_batch['pose'].squeeze(0)
                labelloss = self.adapt_for_labeled_data(learner, h36image, h36s3d, h36s2d, h36beta, h36pose)
                return unlabelloss + labelloss * self.options.labelloss_weight
            else:
                return unlabelloss

    def adapt_for_unlabeled_data(self, learner, image, gt_s2d, hist_image=None, hist_s2d=None, use_consistentLoss=False, only_use_motionloss=False):
        """
        adapt on test data
        """
        batch_size = image.shape[0]
        pred_rotmat, pred_betas, pred_cam = learner(image)

        # convert it to smpl verts and keypoints
        pred_smpl_items = self.decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
        pred_s3ds = pred_smpl_items['s3d']
        pred_vts = pred_smpl_items['vts']

        # project 3d kp to 2d kp
        pred_cam_t = torch.stack([pred_cam[:,1],
                                  pred_cam[:,2],
                                  2*constants.FOCAL_LENGTH/(self.options.img_res * pred_cam[:,0] +1e-9)],dim=-1)
        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_s2d = perspective_projection(pred_s3ds,
                                          rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                          translation=pred_cam_t,
                                          focal_length=constants.FOCAL_LENGTH,
                                          camera_center=camera_center)
        # normalized to [-1,1]
        pred_s2d = pred_s2d / (self.options.img_res / 2.)

        # cal kp2d loss
        s2ds_loss = self.cal_s2ds_loss(pred_s2d, gt_s2d)
        # cal prior loss
        shape_prior_loss = self.shape_prior(pred_betas)
        pose_prior_losses = self.pose_prior(pred_rotmat, pred_betas, gmm_prior=True)
        gmm_prior_loss = pose_prior_losses['gmm']

        loss = s2ds_loss * self.options.s2dsloss_weight +\
               shape_prior_loss * self.options.shapepriorloss_weight +\
               gmm_prior_loss * self.options.gmmpriorloss_weight
        
        if hist_image is not None and hist_s2d is not None:
            pred_hist_rotmat, pred_hist_betas, pred_hist_cam = learner(hist_image)
            pred_hist_smpl_items = self.decode_smpl_params(pred_hist_rotmat, pred_hist_betas, pred_hist_cam, neutral=True)
            pred_hist_s3ds = pred_hist_smpl_items['s3d']
            pred_hist_vts = pred_hist_smpl_items['vts']
            # project 3d kp to 2d kp
            pred_hist_cam_t = torch.stack([pred_hist_cam[:,1],
                                    pred_hist_cam[:,2],
                                    2*constants.FOCAL_LENGTH/(self.options.img_res * pred_hist_cam[:,0] +1e-9)],dim=-1)
            camera_center = torch.zeros(batch_size, 2, device=self.device)
            pred_hist_s2d = perspective_projection(pred_hist_s3ds,
                                          rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                          translation=pred_hist_cam_t,
                                          focal_length=constants.FOCAL_LENGTH,
                                          camera_center=camera_center)
            # normalized to [-1,1]
            pred_hist_s2d = pred_hist_s2d / (self.options.img_res / 2.)
            motion_loss = self.cal_motion_loss(pred_s2d, pred_hist_s2d, gt_s2d, hist_s2d)
            loss = loss + motion_loss * self.options.motionloss_weight

        if use_consistentLoss and self.options.use_meanteacher:
            # cal consistent loss
            ema_rotmat, ema_betas, ema_cam = self.ema_model(image)
            consistent_loss = self.cal_consistent_constrain(pred_rotmat, pred_betas, pred_cam, ema_rotmat, ema_betas, ema_cam)
            loss = loss + consistent_loss * self.options.consistentloss_weight
        return loss
    
    def adapt_for_labeled_data(self, learner, gtimage, gts3d, gts2d, gtbetas, gtpose):
        """
        adapt on source data
        """
        batchsize = gtimage.shape[0]
        # forward
        pred_rotmat, pred_betas, pred_cam = self.model(gtimage)

        # convert it to smpl verts and keypoints
        pred_smpl_items = self.decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
        pred_s3ds = pred_smpl_items['s3d']
        pred_vts = pred_smpl_items['vts']

        # project 3d skeleton to image space, and then rescale to [-1,1] and calculate 2k kp reporjection loss
        pred_cam_t = torch.stack([pred_cam[:,1],
                                  pred_cam[:,2],
                                  2*constants.FOCAL_LENGTH/(self.options.img_res * pred_cam[:,0] +1e-9)],dim=-1)
        camera_center = torch.zeros(batchsize, 2, device=self.device)
        pred_s2d = perspective_projection(pred_s3ds,
                                            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
                                            translation=pred_cam_t,
                                            focal_length=constants.FOCAL_LENGTH,
                                            camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_s2d = pred_s2d / (self.options.img_res / 2.)
        s2ds_loss = self.cal_s2ds_loss(pred_s2d, gts2d)
        s3d_loss = self.cal_s3ds_loss(pred_s3ds, gts3d)
        # smpl loss
        gt_rotmat = batch_rodrigues(gtpose.view(-1,3)).view(-1, 24, 3, 3)
        loss_pose = self.criterion_regr(pred_rotmat, gt_rotmat)
        loss_beta = self.criterion_regr(pred_betas, gtbetas)
        # we use the same setting with SPIN
        loss = s3d_loss * 5. + s2ds_loss * 5 + loss_pose * 1. + loss_beta * 0.001
        return loss


    def test(self, databatch, joint_mapper_gt, joint_mapper_h36m):
        """
        test on arrived data
        """
        if '3dpw' in self.options.dataset_name:
            gt_pose = databatch['oripose']
            gt_betas = databatch['oribeta']
            gender = databatch['gender']

        with torch.no_grad():
            # set model to evaluation mode
            self.model.eval()

            # forward
            oriimages = databatch['oriimg']
            pred_rotamt, pred_betas, pred_cam = self.model(oriimages)
            pred_smpl_out = self.decode_smpl_params(pred_rotamt, pred_betas, pred_cam, neutral=True)
            pred_vts = pred_smpl_out['vts']

            # get 14 gt joints, J_regressor maps mesh to 3D keypoints.
            J_regressor_batch = J_regressor[None, :].expand(pred_vts.shape[0], -1, -1).to(self.device)
            if 'h36m' in self.options.dataset_name or 'mpi-inf' in self.options.dataset_name:
                gt_keypoints_3d = databatch['oripose_3d']
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
            else:
                gt_vertices = smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices_female = smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

            # Get 14 predicted joints from the mesh
            pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vts)
            pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
            pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

            # calculate metrics
            # 1. MPJPE
            error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            
            # 2. PA-MPJPE and PCK
            r_error, pck_error = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), needpck=True, reduction=None)
            return {'mpjpe': error, 'pa-mpjpe': r_error, 'pck': pck_error}


    ##########
    # the following is the loss functions
    ##########

    ## -- motion loss
    def cal_motion_loss(self, pred_kps_t, pred_kps_n, gt_kps_t, gt_kps_n):
        """
        pred_kps_t: (B, 49, 2), at time t
        pred_kps_n: (B, 49, 2), at time t-n
        gt_kps_t  : (B, 49, 3), at time t
        gt_kps_n  : (B, 49, 3), at time t-n
        """
        motion_pred = pred_kps_t[:,25:] - pred_kps_n[:,25:]
        motion_gt = gt_kps_t[:,25:,:-1] - gt_kps_n[:,25:,:-1]
        motion_loss = self.criterion_regr(motion_pred, motion_gt)
        return motion_loss
    ## -- motion loss end

    ## -- consistent loss
    def cal_consistent_constrain(self, pred_rotmat, pred_betas, pred_cam, ema_rotmat, ema_betas, ema_cam):
        batchsize = pred_rotmat.shape[0]
        # convert it to smpl verts and keypoints
        pred_smpl_items = self.decode_smpl_params(pred_rotmat, pred_betas, pred_cam, neutral=True)
        pred_s3ds = pred_smpl_items['s3d']
        pred_vts = pred_smpl_items['vts']

        # convert it to smpl verts and keypoints
        ema_smpl_items = self.decode_smpl_params(ema_rotmat, ema_betas, ema_cam, neutral=True)
        ema_s3ds = ema_smpl_items['s3d']
        ema_vts = ema_smpl_items['vts']

        # project 3d skeleton to image space, and then rescale to [-1,1] and calculate 2k kp reporjection loss
        pred_cam_t = torch.stack([pred_cam[:,1],
                                  pred_cam[:,2],
                                  2*constants.FOCAL_LENGTH/(self.options.img_res * pred_cam[:,0] +1e-9)],dim=-1)
        camera_center = torch.zeros(batchsize, 2, device=self.device)
        pred_s2d = perspective_projection(pred_s3ds,
                                            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
                                            translation=pred_cam_t,
                                            focal_length=constants.FOCAL_LENGTH,
                                            camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_s2d = pred_s2d / (self.options.img_res / 2.)
        # project 3d skeleton to image space, and then rescale to [-1,1] and calculate 2k kp reporjection loss
        ema_cam_t = torch.stack([ema_cam[:,1],
                                  ema_cam[:,2],
                                  2*constants.FOCAL_LENGTH/(self.options.img_res * pred_cam[:,0] +1e-9)],dim=-1)
        camera_center = torch.zeros(batchsize, 2, device=self.device)
        ema_s2d = perspective_projection(ema_s3ds,
                                            rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batchsize, -1, -1),
                                            translation=ema_cam_t,
                                            focal_length=constants.FOCAL_LENGTH,
                                            camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        ema_s2d = ema_s2d / (self.options.img_res / 2.)
        s2ds_loss = self.cal_s2ds_loss_for_mt(pred_s2d, ema_s2d)
        s3d_loss = self.cal_s3ds_loss_for_mt(pred_s3ds, ema_s3ds)
        # smpl loss
        # gt_rotmat = batch_rodrigues(gtpose.view(-1,3)).view(-1, 24, 3, 3)
        loss_pose = self.criterion_regr(pred_rotmat, ema_rotmat)
        loss_beta = self.criterion_regr(pred_betas, ema_betas)
        # loss = s3d_loss * 5. + s2ds_loss * 5 + loss_pose * 1. + loss_beta * 0.001
        loss = s3d_loss * self.options.consistent_s3d_weight + s2ds_loss * self.options.consistent_s2d_weight +\
               loss_pose * self.options.consistent_pose_weight + loss_beta * self.options.consistent_beta_weight
        return loss

    def cal_s3ds_loss_for_mt(self, pred_s3d, gt_s3d):
        """
        pred_s3d: (B, 49, 3)
        gt_s3d: (B, 49, 4)

        """
        # conf = gt_s3d[:,:,-1].unsqueeze(-1).clone()
        gt_s3d = gt_s3d[:,25:]
        pred_s3d = pred_s3d[:,25:]
        # align the root
        gt_hip = (gt_s3d[:,2] + gt_s3d[:,3]) / 2
        gt_s3d = gt_s3d - gt_hip[:,None,:]
        pred_hip = (pred_s3d[:,2] + pred_s3d[:,3]) / 2
        pred_s3d = pred_s3d - pred_hip[:,None,:]
        # print(pred_s3d.shape, gt_s3d.shape, conf.shape)
        loss = (self.criterion_keypoints(pred_s3d, gt_s3d)).mean()
        return loss

    def cal_s2ds_loss_for_mt(self, pred_s2d, gt_s2d):
        """
        pred_s2d: (B, 49, 2)
        gt_s2d: (B, 49, 3)
        only calculate the later 24 joints, i.e., 25:
        """
        # conf = gt_s2d[:,25:,-1].unsqueeze(-1).clone()
        loss = (self.criterion_keypoints(pred_s2d[:,25:], gt_s2d[:,25:])).mean()
        return loss

    ## -- consistent loss end

    def cal_s3ds_loss(self, pred_s3d, gt_s3d):
        """
        pred_s3d: (B, 49, 3)
        gt_s3d: (B, 49, 4)

        """
        conf = gt_s3d[:,:,-1].unsqueeze(-1).clone()
        # gt_s3d = gt_s3d[:,25:]
        pred_s3d = pred_s3d[:,25:]
        # align the root
        gt_hip = (gt_s3d[:,2] + gt_s3d[:,3]) / 2
        gt_s3d = gt_s3d - gt_hip[:,None,:]
        pred_hip = (pred_s3d[:,2] + pred_s3d[:,3]) / 2
        pred_s3d = pred_s3d - pred_hip[:,None,:]
        # print(pred_s3d.shape, gt_s3d.shape, conf.shape)
        loss = (conf * self.criterion_keypoints(pred_s3d, gt_s3d[:,:,:-1])).mean()
        return loss

    def cal_s2ds_loss(self, pred_s2d, gt_s2d):
        """
        pred_s2d: (B, 49, 2)
        gt_s2d: (B, 49, 3)
        only calculate the later 24 joints, i.e., 25:
        """
        conf = gt_s2d[:,25:,-1].unsqueeze(-1).clone()
        loss = (conf * self.criterion_keypoints(pred_s2d[:,25:], gt_s2d[:,25:, :-1])).mean()
        return loss
        
    def shape_prior(self, betas):
        shape_prior_loss = (betas ** 2).sum(dim=-1).mean()
        return shape_prior_loss

    def pose_prior(self, pose, betas, angle_prior=False, gmm_prior=False):
        loss_items = {}
        body_pose = rotation_matrix_to_angle_axis(pose[:,1:].contiguous().view(-1,3,3)).contiguous().view(-1, 69)
        assert body_pose.shape[0] == pose.shape[0]
        if gmm_prior:
            pose_prior_loss = self.criterion_poseprior(body_pose, betas).mean()
            loss_items['gmm'] = pose_prior_loss
        if angle_prior:
            constant = torch.tensor([1., -1., -1, -1.]).to(self.device)
            angle_prior_loss = torch.exp(body_pose[:, [55-3, 58-3, 12-3, 15-3]] * constant) ** 2
            loss_items['angle'] = angle_prior_loss
        return loss_items 
