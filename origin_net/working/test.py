###############################################################
##### @Title:  UWMGI baseline v1
##### @Author: frank
##### @ How to run:
# input:
#  unzip archive.zip => rename "arc" to "uwmgi-mask-dataset"
#  unzip uw-madison-gi-tract-image-segmentation.zip
# working
#  sh train_efficientnet-b1_fold4.sh
##### @Describe: 
#  part0: data preprocess
#  part1: build_transformer() & build_dataset() & build_dataloader()
#  part2: build_model()
#  part3: build_loss()
#  part4: build_metric()
#  part5: train_one_epoch() & valid_one_epoch() & test_one_epoch()
##### @To do: 
#  data: multiresolution...
#  model: resnxt, swin..
#  loss: lovasz, HaussdorfLoss..
#  infer: pseudo labels...
##### @Reference:
# UWMGI: Unet [Train] [PyTorch]: https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/
# UWMGI: Unet [Infer] [PyTorch]: https://www.kaggle.com/code/awsaf49/uwmgi-unet-infer-pytorch/
###############################################################
import os
import pdb
import cv2
import time
import glob
import random

from cv2 import transform
from cv2 import FarnebackOpticalFlow
import cupy as cp  # https://cupy.dev/ => pip install cupy-cuda102
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import torch  # PyTorch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedGroupKFold  # Sklearn
import albumentations as A  # Augmentations
import segmentation_models_pytorch as smp  # smp

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def set_seed(seed=42):
    ##### why 42? The Answer to the Ultimate Question of Life, the Universe, and Everything is 42.
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


###############################################################
##### part0: data preprocess
###############################################################
def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case', ''))
    day = int(data[1].replace('day', ''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case', ''))
    day = int(data[-3].split('_')[1].replace('day', ''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    # row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row


def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk = cp.array(msk)
    pixels = msk.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = [];
    pred_ids = [];
    pred_classes = [];
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(msks[idx],
                         dsize=(width, height),
                         interpolation=cv2.INTER_NEAREST)  # back to original shape
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes


###############################################################
##### part1: build_transformer
###############################################################
# document: https://albumentations.ai/docs/
# example: https://github.com/albumentations-team/albumentations_examples
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            # # dimension should be multiples of 32.
            # ref: https://github.com/facebookresearch/detr/blob/main/datasets/coco.py
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),

            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0] // 20, max_width=CFG.img_size[1] // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms


###############################################################
##### >>>>>>> part1: build_dataset <<<<<<
###############################################################
class build_dataset(Dataset):
    def __init__(self, df, train_label=True, transforms=None):
        self.df = df
        self.train_label = train_label
        self.img_paths = df['image_path'].tolist()
        self.ids = df['id'].tolist()
        if 'mask_path' in df.columns:
            self.msk_paths = df['mask_path'].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        #### id
        id = self.ids[index]
        #### image
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img = np.tile(img[..., None], [1, 1, 3]).astype('float32')  # gray=>rgb
        mx = np.max(img)
        img /= mx  # scale image to [0, 1]

        if self.train_label:  # train
            #### mask
            msk_path = self.msk_paths[index]
            msk = np.load(msk_path)
            msk = msk.astype('float32')
            msk /= 255.0

            ### augmentations
            data = self.transforms(image=img, mask=msk)
            img = data['image']
            msk = data['mask']
            img = np.transpose(img, (2, 0, 1))  # [c, h, w]
            msk = np.transpose(msk, (2, 0, 1))  # [c, h, w]
            return torch.tensor(img), torch.tensor(msk)

        else:  # test
            ### augmentations
            data = self.transforms(image=img)
            img = data['image']
            img = np.transpose(img, (2, 0, 1))  # [c, h, w]
            _, h, w = img.shape
            return torch.tensor(img), id, h, w


###############################################################
##### >>>>>>> part1: build_dataloader <<<<<<
###############################################################
def build_dataloader(df, fold, data_transforms):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)
    train_dataset = build_dataset(train_df, train_label=True, transforms=data_transforms['train'])
    valid_dataset = build_dataset(valid_df, train_label=True, transforms=data_transforms['valid_test'])

    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                              drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


###############################################################
##### >>>>>>> part2: build_model <<<<<<
###############################################################
# document: https://smp.readthedocs.io/en/latest/encoders_timm.html
def build_model(CFG, test_flag=False):
    if test_flag:
        pretrain_weights = None
    else:
        pretrain_weights = "imagenet"
    model = smp.Unet(
        encoder_name=CFG.backbone,
        encoder_weights=pretrain_weights,
        in_channels=3,
        classes=CFG.num_classes,
        activation=None,
    )
    model.to(CFG.device)
    return model


###############################################################
##### >>>>>>> part3: build_loss <<<<<<
###############################################################
def build_loss():
    BCELoss = smp.losses.SoftBCEWithLogitsLoss()
    TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
    return {"BCELoss": BCELoss, "TverskyLoss": TverskyLoss}


###############################################################
##### >>>>>>> part4: build_metric <<<<<<
###############################################################
def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou




@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    val_scores = []

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, masks) in pbar:
        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        masks = masks.to(CFG.device, dtype=torch.float)  # [b, c, w, h]

        y_preds = model(images).sigmoid()  # [b, c, w, h]

        val_dice = dice_coef(masks, y_preds).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_preds).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

    val_scores = np.mean(val_scores, axis=0)
    val_dice, val_jaccard = val_scores
    print("val_dice: {:.4f}, val_jaccard: {:.4f}".format(val_dice, val_jaccard), flush=True)

    return val_dice, val_jaccard


@torch.no_grad()
def test_one_epoch(ckpt_paths, test_loader, CFG):
    pred_strings = []
    pred_ids = []
    pred_classes = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids, h, w) in pbar:
        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        masks = torch.zeros(images.shape, device=CFG.device, dtype=torch.float32)  # [b, c, w, h]

        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################
        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, test_flag=True)
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(images).sigmoid()  # [b, c, w, h]
            masks += y_preds / len(ckpt_paths)

        masks = (masks.permute((0, 2, 3, 1)) > CFG.thr).to(torch.uint8).cpu().detach().numpy()  # [n, h, w, c]
        result = masks2rles(masks, ids, h, w)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
    return pred_strings, pred_ids, pred_classes


if __name__ == '__main__':
    ###############################################################
    ##### >>>>>>> config <<<<<<
    ###############################################################
    class CFG:
        # step1: hyper-parameter
        seed = 20000703
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # step2: data
        n_fold = 4
        img_size = [224, 224]
        train_bs = 64
        valid_bs = 16
        # step3: model

        backbone = 'efficientnet-b3'
        ckpt_name ='efficientnet-b3_img224_64_2_fold4_20220531_23'
        num_classes = 3
        # step4: optimizer
        epoch = 15
        lr = 0.001
        wd = 1e-6
        lr_drop = 10
        # step5: infer
        thr = 0.45
        train_val_flag = False
        test_flag = True
    


    print('--' * 8)
    print('cuda',CFG.device)
    print('backbone', CFG.backbone)
    print('ckpt_name',CFG.ckpt_name)
    print('--' * 8)
    set_seed(CFG.seed)
    # ckpt_path = f"/home/ray/workspace/Fly_Pluche/kaggle/working/ckpt_frank/{CFG.ckpt_name}"
    # ckpt_paths = glob(f'{ckpt_path}/best_fold*.pth')
    ckpt_paths=['/home/ray/workspace/Fly_Pluche/kaggle/working/ckpt_frank/efficientnet-b3_img224_64_2_fold4_20220531_23/best_fold0_epoch18_val_dice0.8931043744087219.pth']
    print('--' * 8)
    # print('ckpt_path:',ckpt_path)
    print('ckpt_paths:', ckpt_paths)
    print('--' * 8)
    assert len(ckpt_paths) > 0, "ckpt path error!"

    set_seed(CFG.seed)
    ###############################################################
    ##### >>>>>>> step1: data prerpocess & dataloader <<<<<<
    ###############################################################
    # "sample_submission.csv" ==> when you click submit, new one will be swaped and populate the text directory.
    # ref: https://www.kaggle.com/code/dschettler8845/uwm-gi-tract-image-segmentation-basic-submission/notebook
    sub_df = pd.read_csv('/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
    if not len(sub_df):
        sub_first = True
        sub_df = pd.read_csv('/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000 * 3]
        sub_df = sub_df.drop(columns=['class', 'segmentation']).drop_duplicates()
        paths = glob(f'/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png', recursive=True)
    else:
        sub_first = False
        sub_df = sub_df.drop(columns=['class', 'predicted']).drop_duplicates()
        paths = glob(f'/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/test/**/*png', recursive=True)

    sub_df = sub_df.apply(get_metadata, axis=1)
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df = path_df.apply(path2info, axis=1)
    test_df = sub_df.merge(path_df, on=['case', 'day', 'slice'], how='left')

    data_transforms = build_transforms(CFG)
    test_dataset = build_dataset(test_df, train_label=False, transforms=data_transforms['valid_test'])
    test_loader = DataLoader(test_dataset, batch_size=CFG.valid_bs, num_workers=2, shuffle=False, pin_memory=False)

    ###############################################################
    ##### >>>>>>> step2: infer <<<<<<
    ###############################################################
    # attention: change the corresponding upload path to kaggle.

    pred_strings, pred_ids, pred_classes = test_one_epoch(ckpt_paths, test_loader, CFG)

    ###############################################################
    ##### step3: submit
    ###############################################################
    pred_df = pd.DataFrame({
        "id": pred_ids,
        "class": pred_classes,
        "predicted": pred_strings
    })
    if not sub_first:
        sub_df = pd.read_csv('/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
        del sub_df['predicted']
    else:
        sub_df = pd.read_csv('/home/ray/workspace/Fly_Pluche/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000 * 3]
        del sub_df['segmentation']

    sub_df = sub_df.merge(pred_df, on=['id', 'class'])
    sub_df.to_csv('submission.csv', index=False)
    print('test over')