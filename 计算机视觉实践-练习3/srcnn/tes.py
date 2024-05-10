import math
import mindspore as ms
from mindspore import nn
import cv2
from dataset import MyData, SRCNNDataset
import mindspore.dataset.transforms as transforms
from mindspore.dataset import vision
import mindspore.dataset as ds
import mysrcnn
import numpy as np
data_test = MyData(r"E:\mindspore\srcnn\Set5\original")


def ssim_compute(out, label):
    mean_out = np.average(out)


HR_test = []
LR_test = []
for data in data_test:
    HR_test.append(data)

for i in range(len(HR_test)):
    LR = cv2.resize(HR_test[i], dsize=None, fx=0.5, fy=0.5)
    LR_test.append(cv2.resize(LR, dsize=None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))


data_trans_list = transforms.Compose([vision.HWC2CHW(), transforms.TypeCast(ms.float32)])
dataset_test = ds.GeneratorDataset(SRCNNDataset(HR_test, LR_test), ["HR", "LR"], shuffle=False)
dataset_test = dataset_test.map(operations=data_trans_list)
dataset_test = dataset_test.map(operations=data_trans_list, input_columns=["LR"])
data_loader = dataset_test.batch(batch_size=1)

network = mysrcnn.SrCnnTest()
param_dict = ms.load_checkpoint("srcnn_norm5_3.ckpt")
param_not_load, _ = ms.load_param_into_net(network, param_dict)
loss_fn = nn.MSELoss(reduction="mean")


def forward_fn(x, label):
    out = network(x)
    loss = loss_fn(out, label)
    psnr = 10 * math.log10(255 ** 2 / loss)
    return psnr


cnt = 0
for data in data_loader:
    img_LR, img_HR = data
    img_LR = ms.ops.expand_dims(img_LR, 1)
    img_HR = ms.ops.expand_dims(img_HR, 1)
    psnr_test = forward_fn(img_LR, img_HR)
    bicubic_psnr = 10 * math.log10(255 ** 2 / loss_fn(img_LR, img_HR))
    cnt = cnt + 1
    print(f"第{cnt}张图片的神经网络输出的PSNR={psnr_test},插值PSNR{bicubic_psnr}")
