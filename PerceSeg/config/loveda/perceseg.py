from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.loveda_dataset import *
from geoseg.models.FTUNetFormer import ft_unetformer
from tools.utils import Lookahead
from tools.utils import process_model_params

# training hparam
max_epoch = 80
ignore_index = len(CLASSES)
train_batch_size = 1
val_batch_size = 1
# lr = 6e-4
lr = 5e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

# weights_name = "ftunetformer-r18-512-crop-ms-e60-base"# 应该是e30
# weights_name = "ftunetformer-r18-512-crop-ms-e60-base-SoftCrossEntropy"# 应该是e30
# weights_name = "ftunetformer-r18-512-crop-ms-e60-base-SoftCrossEntropy-1"# 应该是e30
weights_name = "ftunetformer-r18-512-crop-ms-e60-base-Dice"# 应该是e30
# weights_name = "ftunetformer-r18-512-crop-ms-Dice-mask-2-new"# 应该是e30
# weights_name = "ftunetformer-r18-512-crop-ms-Dice-mask-base-4"# 应该是e30 记得测还剩下456的
# weights_name = "ftunetformer-Dice-mask-base-best"# 应该是e30
# weights_name = "ftunetformer-Dice-mask-base-best-bs4"# 应该是e30
# weights_name = "bs8"

weights_path = "model_weights/loveda/{}".format(weights_name)
# test_weights_name = "ftunetformer-v1"
# test_weights_name = "ftunetformer-r18-512-crop-ms-e60-base"
# test_weights_name = "ftunetformer-r18-512-crop-ms-Dice-mask-base-2-v4"
# test_weights_name = "ftunetformer-r18-512-crop-ms-e60-base-Dice"
test_weights_name = "ftunetformer-r18-512-crop-ms-e60-base-Dice-v3"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 10
save_last = False
check_val_every_n_epoch = 2
# check_val_every_n_epoch = 5
pretrained_ckpt_path = None # the path for the pretrained model weight
# pretrained_ckpt_path = "GeoSeg/pretrain.ckpt" # the path for the pretrained model weight
# pretrained_ckpt_path = "model_weights/loveda/ftunetformer-r18-512-crop-ms-e60-base-Dice/ftunetformer-r18-512-crop-ms-e60-base-Dice-v1.ckpt" # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = ft_unetformer(num_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.5)

use_aux_loss = False

def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

# define the dataloader

train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/Train')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()

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
# base_optimizer = torch.optim.SGD(net_params, lr=lr, momentum=0.9)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

