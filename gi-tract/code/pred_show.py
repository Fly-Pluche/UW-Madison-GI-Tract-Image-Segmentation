from distutils.command.config import config
import os
import argparse
import random
import multiprocessing as mp
from datetime import datetime
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
from tqdm import tqdm

import matplotlib.pylab as plt
import cv2
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torchvision.transforms.functional as F
import torch.utils.data as data
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from yaml import DirectiveToken

from augment import make_train_augmenter
from dataset import VisionDataset
from models import ModelWrapper
from config import Config
import util

cuda_device = '0'

os.environ["CUDA_VISABLE_DEVICES"] = cuda_device

parser = argparse.ArgumentParser()
parser.add_argument('-j',
                    '--num-workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs',
                    default=90,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-p',
                    '--print-interval',
                    default=100,
                    type=int,
                    metavar='N',
                    help='print-interval in batches')
parser.add_argument('--seed',
                    default=20000703,
                    type=int,
                    help='seed for initializing the random number generator')
parser.add_argument(
    '--resume',
    default='',
    # '/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/ckpt/Unet_efficientnet-b5_7_320/bestmodel_Unetefficientnet-b5.pth',
    type=str,
    metavar='PATH',
    help='path to saved model')
parser.add_argument(
    '-s',
    '--subset',
    default=100,
    type=int,
    metavar='N',
    help='use a percentage of the data for training and validation')
parser.add_argument(
    '--input',
    default='/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/input',
    metavar='DIR',
    help='input directory')

device_type = f'cuda:{int(cuda_device)}' if torch.cuda.is_available(
) else 'cpu'
device = torch.device(device_type)


class Trainer:

    def __init__(self,
                 conf,
                 input_dir,
                 device,
                 num_workers,
                 checkpoint,
                 print_interval=100,
                 subset=100):
        self.conf = conf
        self.input_dir = input_dir
        self.device = device
        self.max_patience = 10
        self.print_interval = print_interval
        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()

        self.create_dataloaders(num_workers, subset)

        self.model = ModelWrapper(conf, self.num_classes)
        self.model = self.model.to(device)
        self.optimizer = self.create_optimizer(conf, self.model)
        assert self.optimizer is not None, f'Unknown optimizer {conf.optim}'
        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            # state={
            #     'model': self.model.state_dict()
            # }
            # torch.save(state,'/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/ckpt/Unet_efficientnet-b5_7_320/bestmodel_Unetefficientnet-b5_2.pth')

            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=conf.gamma)
        self.loss_funcs = [
            smp.losses.SoftBCEWithLogitsLoss(),
            smp.losses.TverskyLoss(mode='multilabel', log_loss=False),
        ]
        self.history = None

    def create_dataloaders(self, num_workers, subset):
        conf = self.conf
        meta_file = os.path.join(self.input_dir, 'train.csv')
        assert os.path.exists(
            meta_file), f'{meta_file} not found on Compute Server'
        meta_df = pd.read_csv(meta_file, dtype=str)
        class_names = util.get_class_names(meta_df)
        self.num_classes = len(class_names)

        df = util.process_files(self.input_dir, 'train', meta_df, class_names)
        # shuffle
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        train_aug = make_train_augmenter(conf)
        test_aug = util.make_test_augmenter(conf)

        # split into train and validation sets
        split = df.shape[0] * 90 // 100
        train_df = df.iloc[:split].reset_index(drop=True)
        val_df = df.iloc[split:].reset_index(drop=True)
        train_dataset = VisionDataset(train_df,
                                      conf,
                                      self.input_dir,
                                      'train',
                                      'mask',
                                      class_names,
                                      train_aug,
                                      subset=subset)
        val_dataset = VisionDataset(val_df,
                                    conf,
                                    self.input_dir,
                                    'train',
                                    'mask',
                                    class_names,
                                    test_aug,
                                    subset=subset)
        drop_last = (len(train_dataset) % conf.batch_size) == 1
        self.train_loader = data.DataLoader(train_dataset,
                                            batch_size=conf.batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=False,
                                            worker_init_fn=worker_init_fn,
                                            drop_last=drop_last)
        self.val_loader = data.DataLoader(val_dataset,
                                          batch_size=conf.batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                          pin_memory=False)

    def create_optimizer(self, conf, model):
        if conf.optim == 'sgd':
            return torch.optim.SGD(model.parameters(),
                                   lr=conf.lr,
                                   momentum=0.9,
                                   weight_decay=conf.weight_decay)
        if conf.optim == 'adamw':
            return torch.optim.AdamW(model.parameters(),
                                     lr=conf.lr,
                                     weight_decay=conf.weight_decay)
        return None

    def fit(self, epochs):
        best_val_score = None
        patience = self.max_patience
        self.sample_count = 0
        self.history = util.LossHistory()

        print(f'Running on {device}')
        print(f'{len(self.train_loader.dataset)} examples in training set')
        print(f'{len(self.val_loader.dataset)} examples in validation set')
        trial = os.environ.get('TRIAL')
        suffix = f"-trial{trial}" if trial is not None else ""
        log_dir = f"runs/{datetime.now().strftime('%b_%d_%H_%M_%S')}{suffix}_{self.conf.arch}{self.conf.backbone}"
        writer = SummaryWriter(log_dir=log_dir)

        print('Training in progress...')
        save_path = f'/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/ckpt/{self.conf.arch}_{self.conf.backbone}_{self.conf.in_channels}_{self.conf.image_size}'
        if os.path.isdir(save_path) == 0:
            os.mkdir(save_path)
        for epoch in range(epochs):
            # train for one epoch
            print(f'Epoch {epoch}:')
            # train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()
            self.scheduler.step()
            # writer.add_scalar('loss/train_loss', train_loss, epoch)
            writer.add_scalar('loss/val_loss', val_loss, epoch)
            writer.add_scalar('val_score', val_score, epoch)

            self.history.save()
        # writer.close()
        writer.flush()
        return '{save_path}/bestmodel_{self.conf.arch}{self.conf.backbone}.pth'

    def criterion(self, outputs, labels):
        result = 0
        for func in self.loss_funcs:
            result += func(outputs, labels)
        return result

    def train_epoch(self, epoch):
        model = self.model
        optimizer = self.optimizer

        # val_iter = iter(self.val_loader)
        # val_interval = len(self.train_loader)//len(self.val_loader)
        # assert val_interval > 0
        train_loss_list = []
        model.train()
        pdar = tqdm(enumerate(self.train_loader),
                    total=len(self.train_loader),
                    desc='train')
        for step, (images, labels) in pdar:

            images = images.to(device)
            labels = labels.to(device)
            # compute output
            # use AMP
            with autocast(enabled=self.use_amp):
                outputs = model(images)
                loss = self.criterion(outputs, labels)

            train_loss_list.append(loss.item())
            self.sample_count += images.shape[0]
            self.history.add_train_loss(epoch, self.sample_count, loss.item())
            # if (step + 1) % self.print_interval == 0:
            #     print(f'Batch {step + 1}: training loss {loss.item():.5f}')
            # compute gradient and do SGD step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()

        mean_train_loss = np.array(train_loss_list).mean()
        return

    def validate(self):
        sigmoid = nn.Sigmoid()
        losses = []
        scores = []
        self.model.eval()
        save_path = '/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/pred/'
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader),
                        total=len(self.val_loader),
                        desc='eval')
            j = 0
            for index, (images, labels) in pbar:
                images = images.to(device)
                labels = labels.to(device)
                with autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                preds = sigmoid(outputs).round().to(torch.float32)
                scores.append(util.dice_coeff(labels, preds).item())
                losses.append(self.criterion(outputs, labels).item())
                for i in range(self.conf.batch_size):
                    a = preds[i].cpu().numpy().transpose(1, 2, 0)
                    b = labels[i].cpu().numpy().transpose(1, 2, 0)
                    img = np.expand_dims(
                        images.cpu().numpy()[1].transpose(
                            1, 2, 0)[:, :, self.conf.in_channels // 2],
                        2).repeat(3, 2)
                    c = np.concatenate((img, a, b), axis=1)
                    cv2.imwrite(save_path + str(j) + '.png', c * 255)
                    j += 1
        return np.mean(losses), np.mean(scores)


def worker_init_fn(worker_id):
    random.seed(random.randint(0, 2**32) + worker_id)
    np.random.seed(random.randint(0, 2**32) + worker_id)


def main():
    args = parser.parse_args()
    if args.subset != 100:
        print(
            f'\nWARNING: {args.subset}% of the data will be used for training\n'
        )
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    input_dir = args.input
    model_file = args.resume
    if model_file:
        print(f'Loading model from {model_file}')
        checkpoint = torch.load(model_file)
        # conf = checkpoint['conf']
    else:
        checkpoint = None
    conf = Config()

    trainer = Trainer(conf, input_dir, device, args.num_workers, checkpoint,
                      args.print_interval, args.subset)
    trainer.fit(args.epochs)


if __name__ == '__main__':
    main()