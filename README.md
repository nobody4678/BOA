## Getting started
To run on 3DPW, you need to run:
```python
python boa.py --expdir exp --name 3dpw-exp --use_mixtrain --labelloss_weight 0.1 --use_meanteacher --ema_decay 0.1\
              --consistentloss_weight 0.1 --use_maml --use_motionloss --metalr 8e-6 --motionloss_weight 0.1
```

Here are critical requirements of installation:
```buildoutcfg
neural-renderer-pytorch
numpy
opencv-python
pyopengl
pyrender
scikit-image
scipy==1.0.0
tensorboard
chumpy
smplx
spacepy
torch==1.1.0
torchgeometry
torchvision
tqdm
trimesh
learn2learn
joblib
```

Pretrained model and datasets will be released later.

