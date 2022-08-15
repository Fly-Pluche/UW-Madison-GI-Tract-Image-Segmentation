# %%
# !pip install -q ../input/wheels/pretrainedmodels-0.7.4-py3-none-any.whl
# !pip install -q ../input/wheels/efficientnet_pytorch-0.6.3-py3-none-any.whl
# !pip install -q ../input/wheels/timm-0.4.12-py3-none-any.whl
# !pip install -q ../input/wheels/segmentation_models_pytorch-0.2.1-py3-none-any.whl

# %%
# !pip -q wheel segmentation_models_pytorch
import os
import glob
import torch
import PIL
import torchvision.transforms.functional as F
from tqdm import tqdm
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.utils.data as data

import numpy as np
import cv2


# %%
def get_class_names(df):
    labels = df['class']
    return labels.unique()


def make_test_augmenter(conf):
    crop_size = round(conf.image_size * conf.crop_size)
    return A.Compose([
        A.CenterCrop(height=crop_size, width=crop_size),
        ToTensorV2(transpose_mask=True)
    ])


def get_id(filename):
    # e.g. filename: case123_day20/scans/slice_0001_266_266_1.50_1.50.png
    # id: case123_day20_slice_0001
    tokens = filename.split('/')
    return tokens[-3] + '_' + '_'.join(tokens[-1].split('_')[:2])


class VisionDataset(data.Dataset):

    def __init__(self,
                 df,
                 conf,
                 input_dir,
                 imgs_dir,
                 class_names,
                 transform,
                 is_test=False,
                 subset=100):
        self.conf = conf
        self.transform = transform
        self.is_test = is_test

        if subset != 100:
            assert subset < 100
            # train and validate on subsets
            num_rows = df.shape[0] * subset // 100
            df = df.iloc[:num_rows]

        files = df['img_files']
        self.files = [os.path.join(input_dir, imgs_dir, f) for f in files]
        self.masks = [f.replace('train', 'masks') for f in files]

    def resize(self, img, interp):
        return cv2.resize(img, (self.conf.image_size, self.conf.image_size),
                          interpolation=interp)

    def load_slice(self, img_file, diff):
        slice_num = os.path.basename(img_file).split('_')[1]
        filename = (img_file.replace(
            'slice_' + slice_num,
            'slice_' + str(int(slice_num) + diff).zfill(4)))
        if os.path.exists(filename):
            return cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        return None

    def __getitem__(self, index):
        conf = self.conf
        img_file = self.files[index]
        mid = conf.in_channels // 2
        # read 5 slices into one image
        imgs = [self.load_slice(img_file, i) for i in range(-mid, mid + 1)]
        for i in range(mid, conf.in_channels - 1):
            if imgs[i + 1] is None:
                imgs[i + 1] = imgs[i]
        for i in range(mid, 0, -1):
            if imgs[i - 1] is None:
                imgs[i - 1] = imgs[i]

        img = np.stack(imgs, axis=2)
        img = img.astype(np.float32)
        max_val = img.max()
        if max_val != 0:
            img /= max_val
        img = self.resize(img, cv2.INTER_AREA)

        if self.is_test:
            msk = 0
            result = self.transform(image=img)
            img = result['image']
        else:
            # read mask
            msk_file = self.masks[index]
            msk = cv2.imread(msk_file, cv2.IMREAD_UNCHANGED)
            msk = self.resize(msk, cv2.INTER_NEAREST)
            msk = msk.astype(np.float32)
            result = self.transform(image=img, mask=msk)
            img, msk = result['image'], result['mask']
        return img, msk

    def __len__(self):
        return len(self.files)


class ModelWrapper(nn.Module):

    def __init__(self, conf, num_classes):
        super().__init__()
        if conf.arch == 'FPN':
            arch = smp.FPN
        elif conf.arch == 'Unet':
            arch = smp.Unet
        elif conf.arch == 'DeepLabV3':
            arch = smp.DeepLabV3
        else:
            assert 0, f'Unknown architecture {conf.arch}'

        weights = 'imagenet' if conf.pretrained else None
        self.model = arch(encoder_name=conf.backbone,
                          encoder_weights=weights,
                          in_channels=conf.in_channels,
                          classes=num_classes,
                          activation=None)

    def forward(self, x):
        x = self.model(x)
        return x


# %%


def create_test_loader(conf, input_dir, class_names):
    test_aug = make_test_augmenter(conf)
    test_df = pd.DataFrame()
    img_files = []
    img_dir = 'test'
    subdir = ''
    while len(img_files) == 0 and len(subdir) < 10:
        img_files = sorted(glob.glob(f'{input_dir}/{img_dir}/{subdir}*.png'))
        subdir += '*/'
        if len(subdir) > 10:
            return None
    # delete common prefix from paths
    if len(img_files) == 0:
        img_dir = 'train'
        subdir = ''
        while len(img_files) == 0 and len(subdir) < 10:
            img_files = sorted(
                glob.glob(f'{input_dir}/{img_dir}/{subdir}*.png'))
            subdir += '*/'
            if len(subdir) > 10:
                return None
        img_files = img_files[:1000]
    img_files = [f.replace(f'{input_dir}/{img_dir}/', '') for f in img_files]

    test_df['img_files'] = img_files
    test_dataset = VisionDataset(test_df,
                                 conf,
                                 input_dir,
                                 img_dir,
                                 class_names,
                                 test_aug,
                                 is_test=True)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(test_dataset,
                             batch_size=conf.batch_size,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=False)
    return loader, test_df


def create_model(conf, model_dir, num_classes):
    path_list = glob.glob(model_dir)
    path = path_list[0]
    print('path_list', path_list)
    assert len(path_list) != 0
    checkpoint = torch.load(path, map_location=device)
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model


def rle_encode(img):
    '''
    this function is adapted from
    https://www.kaggle.com/code/stainsby/fast-tested-rle/notebook
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_img_shape(filename):
    basename = os.path.basename(filename)
    tokens = basename.split('_')
    height, width = int(tokens[3]), int(tokens[2])
    return (height, width)


def pad_mask(conf, mask):
    # pad image to conf.image_size
    padded = np.zeros((conf.image_size, conf.image_size), dtype=mask.dtype)
    dh = conf.image_size - mask.shape[0]
    dw = conf.image_size - mask.shape[1]

    top = dh // 2
    left = dw // 2
    padded[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
    return padded


def resize_mask(mask, height, width):
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


# %%


def df_preprocessing(df, globbed_file_list, is_test=False):
    """ The preprocessing steps applied to get column information """
    # 1. Get Case-ID as a column (str and int)
    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[0].replace("case", "")))

    # 2. Get Day as a column
    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(
        lambda x: int(x.split("_", 2)[1].replace("day", "")))

    # 3. Get Slice Identifier as a column
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    # 4. Get full file paths for the representative scans
    df["_partial_ident"] = (
        globbed_file_list[0].rsplit("/", 4)[0] +
        "/" +  # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
        df["case_id_str"] + "/" +  # .../case###/
        df["case_id_str"] + "_" + df["day_num_str"] +  # .../case###_day##/
        "/scans/" + df["slice_id"])  # .../slice_####
    _tmp_merge_df = pd.DataFrame({
        "_partial_ident": [x.rsplit("_", 4)[0] for x in globbed_file_list],
        "f_path":
        globbed_file_list
    })
    df = df.merge(_tmp_merge_df,
                  on="_partial_ident").drop(columns=["_partial_ident"])

    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df["f_path"].apply(
        lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["f_path"].apply(
        lambda x: float(x[:-4].rsplit("_", 4)[4]))

    if not is_test:
        # 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
        l_bowel_df = df[df["class"] == "large_bowel"][[
            "id", "segmentation"
        ]].rename(columns={"segmentation": "lb_seg_rle"})
        s_bowel_df = df[df["class"] == "small_bowel"][[
            "id", "segmentation"
        ]].rename(columns={"segmentation": "sb_seg_rle"})
        stomach_df = df[df["class"] == "stomach"][[
            "id", "segmentation"
        ]].rename(columns={"segmentation": "st_seg_rle"})
        df = df.merge(l_bowel_df, on="id", how="left")
        df = df.merge(s_bowel_df, on="id", how="left")
        df = df.merge(stomach_df, on="id", how="left")
        df = df.drop_duplicates(subset=[
            "id",
        ]).reset_index(drop=True)
        df["lb_seg_flag"] = df["lb_seg_rle"].apply(lambda x: not pd.isna(x))
        df["sb_seg_flag"] = df["sb_seg_rle"].apply(lambda x: not pd.isna(x))
        df["st_seg_flag"] = df["st_seg_rle"].apply(lambda x: not pd.isna(x))
        df["n_segs"] = df["lb_seg_flag"].astype(int) + df[
            "sb_seg_flag"].astype(int) + df["st_seg_flag"].astype(int)

    # 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
    new_col_order = [
        "id",
        "f_path",
        "n_segs",
        "lb_seg_rle",
        "lb_seg_flag",
        "sb_seg_rle",
        "sb_seg_flag",
        "st_seg_rle",
        "st_seg_flag",
        "slice_h",
        "slice_w",
        "px_spacing_h",
        "px_spacing_w",
        "case_id_str",
        "case_id",
        "day_num_str",
        "day_num",
        "slice_id",
    ]
    if is_test: new_col_order.insert(1, "class")
    new_col_order = [_c for _c in new_col_order if _c in df.columns]
    df = df[new_col_order]

    return df


def fix_empty_slices(_row):
    if int(_row["slice_id"].rsplit("_",
                                   1)[-1]) in remove_seg_slices[_row["class"]]:
        _row["predicted"] = ""
    return _row


def is_isolated(_row):
    return (_row["predicted"] != "" and _row["prev_predicted"] == ""
            and _row["next_predicted"] == "")


def fix_nc_slices(_row):
    if _row["seg_isolated"]:
        _row["predicted"] = ""
    return _row


remove_seg_slices = {
    "large_bowel": [1, 138, 139, 140, 141, 142, 143, 144],
    "small_bowel":
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 138, 139, 140, 141, 142, 143, 144],
    "stomach": [
        1, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 143, 144
    ],
}


def run(input_dir, model_dir, thresh):
    meta_file = os.path.join(input_dir, 'train.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)

    model = create_model(conf, model_dir, num_classes)
    loader, df = create_test_loader(conf, input_dir, class_names)
    img_files = df['img_files']

    subm = pd.read_csv(f'{input_dir}/sample_submission.csv')
    del subm['predicted']

    ids = []
    classes = []
    masks = []
    img_idx = 0
    sigmoid = nn.Sigmoid()
    model.eval()
    with torch.no_grad():
        #         qdar=tqdm(enumerate(loader),le=len())
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            preds = sigmoid(outputs).cpu().numpy()

            # if conf.tta:
            #     flips = [[-1], [-2], [-2, -1]]
            #     for f in flips:
            #         images_f = torch.flip(images, f)
            #         y_preds = model(images_f)
            #         y_preds = torch.flip(y_preds, f)
            #         y_preds = torch.nn.Sigmoid()(y_preds)
            #         preds = np.sum((y_preds, preds), axis=0)

            #     preds /= 4
            preds[preds >= thresh] = 1
            preds[preds < thresh] = 0
            for pred in preds:
                img_file = img_files[img_idx]
                img_idx += 1
                img_id = get_id(img_file)
                height, width = get_img_shape(img_file)
                for class_id, class_name in enumerate(class_names):
                    mask = pred[class_id]
                    mask = pad_mask(conf, mask)
                    mask = resize_mask(mask, height, width)
                    enc_mask = '' if mask.sum() == 0 else rle_encode(mask)
                    ids.append(img_id)
                    classes.append(class_name)
                    masks.append(enc_mask)

        ss_df = pd.DataFrame({'id': ids, 'class': classes, 'predicted': masks})
        if ss_df.shape[0] > 0:
            # sort according to the given order and save to a csv file
            ss_df = ss_df.merge(subm, on=['id', 'class'])

        # Get all testing images if there are any
        subm.to_csv("submission.csv", index=False)
        return subm


# %%
class Config():
    # FPN, Unet or DeepLab
    arch = 'Unet'
    backbone = 'efficientnet-b3'
    pretrained = True
    in_channels = 7
    # crop to this fraction of image_size
    crop_size = 0.9
    # resize images to this size on the fly
    image_size = 320

    # optimizer settings
    optim = 'adamw'
    lr = 0.001
    weight_decay = 0.01
    batch_size = 1

    # scheduler settings
    gamma = 0.96

    # data augmentation
    aug_prob = 0.4
    strong_aug = True
    max_cutout = 0

    att = True


conf = Config()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device=torch.device('cpu')
DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
TEST_DIR = os.path.join(DATA_DIR, "test")
# device = torch.device('cpu')

# %%
import os
import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2

test_thresh = 0.45
run(
    '/home/ray/workspace/Fly_Pluche/kaggle/origin_net/input/uw-madison-gi-tract-image-segmentation',
    '/home/ray/workspace/Fly_Pluche/kaggle/gi-tract/ckpt/Unet_efficientnet-b3/bestmodel_Unetefficientnet-b3_*.pth',
    test_thresh)
print('over')

# %% [markdown]
#
