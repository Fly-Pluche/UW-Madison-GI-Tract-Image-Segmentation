
# 【函数名称】：融合显示函数（visualizationShowFusion）
# 【函数功能】：用来显示当前的分割效果
# 【输入参数】：原图（x）torch；预测值（y_pred）torch；实际标签（label）torch；窗口名称（win_name）str；模型输入通道数（input_chennels）int
# 【输出参数】：无
def visualizationShowFusion(x, y_pred, label, win_name, input_chennels,show):
    # 【函数名称】：显示图像归一化函数（MaskimgNormalization）
    # 【函数功能】：将像素值不为0~255的像素全部转化为数值为0~255的像素
    # 【输入参数】：旧图像矩阵（OldImg）
    # 【输出参数】：新图像矩阵（NewImg）
    def MaskimgNormalization(OldImg):
        OldMin = 0.0#OldImg.min()
        OldMax = 1.0#OldImg.max()
        NewMin = 0
        NewMax = 255
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        NewImg = ((((OldImg - OldMin) * NewRange) / OldRange) + NewMin)
        NewImg = NewImg.type(torch.uint8)
        return NewImg
    # 【函数名称】：显示图像归一化函数（CTimgNormalization）
    # 【函数功能】：将像素值不为0~255的像素全部转化为数值为0~255的像素
    # 【输入参数】：旧图像矩阵（OldImg）
    # 【输出参数】：新图像矩阵（NewImg）
    def CTimgNormalization(OldImg):
        OldMin = OldImg.min()
        OldMax = OldImg.max()
        NewMin = 0
        NewMax = 255
        OldRange = (OldMax - OldMin)
        NewRange = (NewMax - NewMin)
        NewImg = ((((OldImg - OldMin) * NewRange) / OldRange) + NewMin)
        NewImg = NewImg.type(torch.uint8)
        return NewImg

    if isinstance(y_pred,tuple):
        y_pred = y_pred[-1]
    if y_pred.dim() == 5:
        x = x[0]
        x = x.permute(1, 0, 2, 3)
        y_pred = y_pred[0]
        y_pred = y_pred.permute(1,0,2,3)
        label = label[0]
        label = label.permute(1, 0, 2, 3)

    y_pred = F.sigmoid(y_pred)

    y_pred_colors = [[0,0,0],[255, 0, 0], [0, 255, 0],[0, 0, 255], [152, 0, 0], [255, 0, 255], [255, 153, 0],[20,40,80],
                     [0, 255, 255], [74, 134, 232], [255, 255, 0], [153, 0, 255],[255,50,155],[50,200,150],[50,50,50],[200,200,100]]
    random_slice = random.randint(0, len(x) - 1)

    x = x[random_slice]
    y_pred = y_pred[random_slice]
    label = label[random_slice]
    # 首先找出原始图像的那一张
    x_mask = CTimgNormalization(x[int(input_chennels / 2)])
    x_mask = np.array(x_mask.tolist(), dtype=np.uint8)
    x_mask = np.stack((x_mask, x_mask, x_mask), axis=0)  # 增加维度
    x_mask = x_mask.transpose(1, 2, 0)  # 交换维度

    for i in range(len(label)):
        y_pred_mask = MaskimgNormalization(y_pred[i])
        label_mask = MaskimgNormalization(label[i])

        y_pred_mask_r = np.array((y_pred_mask * (y_pred_colors[i][0] / 255)).tolist(), dtype=np.uint8)
        y_pred_mask_g = np.array((y_pred_mask * (y_pred_colors[i][1] / 255)).tolist(), dtype=np.uint8)
        y_pred_mask_b = np.array((y_pred_mask * (y_pred_colors[i][2] / 255)).tolist(), dtype=np.uint8)
        label_mask_r = np.array((label_mask * (y_pred_colors[i][0] / 255)).tolist(), dtype=np.uint8)
        label_mask_g = np.array((label_mask * (y_pred_colors[i][1] / 255)).tolist(), dtype=np.uint8)
        label_mask_b = np.array((label_mask * (y_pred_colors[i][2] / 255)).tolist(), dtype=np.uint8)

        y_pred_mask = np.stack((y_pred_mask_b, y_pred_mask_g, y_pred_mask_r), axis=0)
        y_pred_mask = y_pred_mask.transpose(1, 2, 0)
        label_mask = np.stack((label_mask_b, label_mask_g, label_mask_r), axis=0)
        label_mask = label_mask.transpose(1, 2, 0)

        alpha = 1
        beta = 0.5

        if i == 0:
            # 图像融合
            x_y_pred = cv2.addWeighted(x_mask, alpha, y_pred_mask, beta, 0)
            x_label = cv2.addWeighted(x_mask, alpha, label_mask, beta, 0)
        else:
            x_y_pred = cv2.addWeighted(x_y_pred, alpha, y_pred_mask, beta, 0)
            x_label = cv2.addWeighted(x_label, alpha, label_mask, beta, 0)

    spliced_image = np.hstack((x_y_pred, x_label))

    if show == True:
        cv2.imshow(win_name, spliced_image)
        cv2.waitKey(1)

    return spliced_image




#调用
visualizationShowFusion(images, y_preds, masks, "show", input_chennels=3, show=True)
