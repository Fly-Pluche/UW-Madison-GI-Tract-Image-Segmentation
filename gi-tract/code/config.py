import os
import yaml


class Config():

    def __init__(self):
        # FPN, Unet or DeepLab
        self.pretrained = True
        self.backbone = 'efficientnet-b3'
        #https://smp.readthedocs.io/en/latest/encoders.html#resnest
        self.arch = 'Unet'
        #"Unet"
        self.in_channels = 5
        self.fuse = False
        # crop to this fraction of image_size
        self.crop_size = 1
        # resize images to this size on the fly
        self.image_size = 320

        # optimizer settings
        self.optim = 'adamw'
        self.lr = 0.01
        self.weight_decay = 0.001
        self.batch_size = 18

        # scheduler settings
        self.gamma = 0.96

        # loss
        self.loss_3d = False

        # data augmentation
        self.aug_prob = 0.4
        self.strong_aug = True
        self.max_cutout = 0

        # inference
        self.thre = 0.45
        self.tta = True


if __name__ == '__main__':
    conf = Config()
    print(conf.arch)
    print(conf.__dict__)
