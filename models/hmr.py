import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from utils.geometry import rot6d_to_rotmat
from models.nonlocal_lib.non_local_embedded_gaussian import NONLocalBlock2D


"""
To use adaptator, we will try two kinds of schemes.
"""

def gn_helper(planes):
    if 0:
        return nn.BatchNorm2d(planes)
    else:
        # print('use GN')
        return nn.GroupNorm(32 // 8, planes)


class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer=gn_helper, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.adapter_conv1x1_1 = nn.Conv2d(inplanes, planes, kernel_size=1,bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.adapter_conv1x1_2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.adapter_conv1x1_3 = nn.Conv2d(planes, planes * 4, kernel_size=1,bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = norm_layer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # y1 = self.adapter_conv1x1_1(x)
        # out = out + y1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # y2 = self.adapter_conv1x1_2(out)
        # out = out2 + y2
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # y3 = self.adapter_conv1x1_3(out)
        # out = out3 + y3
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HMR(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, non_local=False):
        norm_layer = gn_helper
        self.inplanes = 64
        super(HMR, self).__init__()
        npose = 24 * 6
        # self.non_local = non_local
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.adapter_conv1x1_prev = nn.Conv2d(3,64, kernel_size=1, stride=2, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(norm_layer, block, 64, layers[0])
        self.layer2 = self._make_layer(norm_layer, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(norm_layer, block, 256, layers[2], stride=2, non_local=non_local)
        self.layer4 = self._make_layer(norm_layer, block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion + npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()
        self.decpose = nn.Linear(1024, npose)
        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def _make_layer(self, norm_layer, block, planes, blocks, stride=1, non_local=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, norm_layer, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer))
        
        # add non-local block here
        if non_local:
            layers.append(NONLocalBlock2D(in_channels=self.inplanes))
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):
        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)
        x = self.conv1(x)
        # y2 = self.adapter_conv1x1_prev(x)
        # x = y + y2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam


class HMR_ISO(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params, non_local=False):
        self.inplanes = 64
        super(HMR_ISO, self).__init__()
        npose = 24 * 6
        # self.non_local = non_local
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, non_local=non_local)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.ssl_head = nn.Sequential(nn.Linear(512 * block.expansion + npose + 13, 1024),
                                      nn.Dropout(),
                                      nn.Linear(1024, 1024),
                                      nn.Dropout())
        self.ssl_decpose = nn.Linear(1024, npose)
        self.ssl_decshape = nn.Linear(1024, 10)
        self.ssl_deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.ssl_decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.ssl_decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.ssl_deccam.weight, gain=0.01)

        self.fsl_head = nn.Sequential(nn.Linear(512 * block.expansion + npose + 13, 1024),
                                      nn.Dropout(),
                                      nn.Linear(1024, 1024),
                                      nn.Dropout())
        self.fsl_decpose = nn.Linear(1024, npose)
        self.fsl_decshape = nn.Linear(1024, 10)
        self.fsl_deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.fsl_decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fsl_decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fsl_deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1, non_local=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        # add non-local block here
        if non_local:
            layers.append(NONLocalBlock2D(in_channels=self.inplanes))
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def fsl(self, init_pose, init_shape, init_cam, xf, n_iter, batch_size):
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.fsl_head(xc)
            pred_pose = self.fsl_decpose(xc) + pred_pose
            pred_shape = self.fsl_decshape(xc) + pred_shape
            pred_cam = self.fsl_deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_rotmat, pred_shape, pred_cam
    
    def ssl(self, init_pose, init_shape, init_cam, xf, n_iter, batch_size):
        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        for i in range(n_iter):
            xc = torch.cat([xf, pred_pose, pred_shape, pred_cam],1)
            xc = self.ssl_head(xc)
            pred_pose = self.ssl_decpose(xc) + pred_pose
            pred_shape = self.ssl_decshape(xc) + pred_shape
            pred_cam = self.ssl_deccam(xc) + pred_cam
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)
        return pred_rotmat, pred_shape, pred_cam

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        pred_shape = init_shape
        pred_cam = init_cam
        pred_rotmat_ssl, pred_shape_ssl, pred_cam_ssl = self.ssl(init_pose, init_shape, init_cam, xf, n_iter, batch_size)
        pred_rotmat_fsl, pred_shape_fsl, pred_cam_fsl = self.fsl(init_pose, init_shape, init_cam, xf, n_iter, batch_size)
        return pred_rotmat_fsl, pred_shape_fsl, pred_cam_fsl, pred_rotmat_ssl, pred_shape_ssl, pred_cam_ssl


def hmr(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model