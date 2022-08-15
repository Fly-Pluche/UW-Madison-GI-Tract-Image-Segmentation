import os
import glob
from statistics import mode

import torch


def SWA(model_dir, name_with, save_dir=None):
    model_dirs = sorted(glob.glob(model_dir + name_with))
    # index=model_dirs.index('E:/BaiduNetdiskDownload\\139-model.pth')
    model_dirs = model_dirs[3:8]
    print(len(model_dirs))
    print(model_dirs)
    models = [
        torch.load(model_dir, map_location='cpu') for model_dir in model_dirs
    ]  # 加载网络模型
    model_num = len(models)  # 模型数
    model_keys = models[-1]['model'].keys()  # 模型关键字
    model = models[-1]['model']  #
    new_model = model.copy()
    ref_model = models[-1]  # 新的网络模型初始化

    # swa
    for key in model_keys:
        sum_weight = 0.0
        for m in models:
            sum_weight += m['model'][key]
        avg_weight = sum_weight / model_num
        new_model[key] = avg_weight
    ref_model['model'] = new_model  # 随机权重平均后的权重
    save_model_name = '43_47' + '.pth'
    if save_dir is not None:
        save_dir = os.path.join(save_dir, save_model_name)
    else:
        save_dir = os.path.join(model_dir, save_model_name)
    torch.save(ref_model, save_dir)  # 保存网络权重
    print('Model is saved at', save_dir)


if __name__ == '__main__':
    model_dir = r'E:/BaiduNetdiskDownload/'
    name_with = '4*.pth'
    SWA(model_dir, name_with)