from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
# max_epoch = 65
max_epoch = 45
ignore_index = len(CLASSES)
# train_batch_size = 8
# val_batch_size = 4
train_batch_size = 2
val_batch_size = 1
lr = 6e-4
# lr = 5e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES

# weights_name = "ftunetformer-512-ms-crop-mask-train-lro-mIOU"
# weights_name = "ftunetformer-512-ms-crop-mask"
# weights_name = "ftunetformer-512-ms-crop-mask-mIOU-notrain_mode-75"
# weights_name = "ftunetformer-test"
# weights_name = "ftunetformer-graph-mask"
# weights_name = "test-graph-mask-clusters"
weights_name = "test-mask/0.25"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
# test_weights_name = "ftunetformer-512-ms-crop-mask-mIOU-notrain_mode-75"
test_weights_name = "test-mask/0.25"
log_name = 'vaihingen/{}'.format(weights_name)
# monitor = 'val_F1'
monitor = 'val_mIoU'
# monitor = 'val_OA'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = "model_weights/vaihingen/ftunetformer-512-ms-crop/ftunetformer-512-ms-crop-v8.ckpt" # the path for the pretrained model weight
# pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = ft_unetformer(num_classes=num_classes, decoder_channels=256)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = False

# define the dataloader

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)



