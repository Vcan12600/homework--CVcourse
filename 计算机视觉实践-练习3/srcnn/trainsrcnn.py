from dataset import dataset
import mysrcnn
import mindspore as ms
from mindspore import nn
import mindspore.dataset as ds
from mindspore.dataset import vision
import mindspore.dataset.transforms as transforms
import math
from matplotlib import pyplot as plt
import numpy as np

# 模型加载demo
# model = network()
# param_dict = mindspore.load_checkpoint("model.ckpt")
# param_not_load, _ = mindspore.load_param_into_net(model, param_dict)


ms.set_context(mode=ms.context.GRAPH_MODE, device_target="CPU")
save_path = './srcnn_norm5_3.ckpt'
dataset = dataset
data_trans_list = transforms.Compose([vision.HWC2CHW(), transforms.TypeCast(ms.float32)])
dataset_train = ds.GeneratorDataset(dataset, ["HR", "LR"], shuffle=True)
dataset_train = dataset_train.map(operations=data_trans_list)
dataset_train = dataset_train.map(operations=data_trans_list, input_columns=["LR"])
data_loader = dataset_train.batch(batch_size=4)  # input_columns


network = mysrcnn.SrCnn()
loss_fn = nn.MSELoss(reduction="mean")
# optimizer = ms.nn.Momentum(params=network.trainable_params(), learning_rate=0.0001, momentum=0.9)
optimizer = ms.nn.Adam(params=network.trainable_params(), learning_rate=0.0003)
param_dict = ms.load_checkpoint("srcnn_norm5_3.ckpt")  # best 45
param_not_load, _ = ms.load_param_into_net(network, param_dict)


def forward_fn(x, label):
    out = network(x)
    loss = loss_fn(out, label)
    return loss


grad_fn = ms.value_and_grad(forward_fn, grad_position=None, weights=network.trainable_params())


def train_step(data, label):
    (loss, grads) = grad_fn(data, label)
    optimizer(grads)
    return loss


loss_list = [0, 0]
for i in range(40):
    cnt = 0
    sum_loss = 0
    sum_psnr = 0
    cnt_step = 0

    for img_data in data_loader:
        LR, HR = img_data
        LR = ms.ops.expand_dims(LR, 1)
        HR = ms.ops.expand_dims(HR, 1)
        mse = train_step(LR, HR)
        mse = mse/4
        psnr = 10*math.log10(255**2/mse)
        sum_loss = sum_loss + mse
        sum_psnr = sum_psnr + psnr
        cnt = cnt + 1
        percent = cnt/len(data_loader)
        if cnt % 1000 == 0:
            cnt_step = cnt_step+1
            print(f"{i+1}epoch数据集进度{round(percent*100, 2)}%,Loss={sum_loss/(1000*cnt_step)},"
                  f"psnr={sum_psnr/(1000*cnt_step)}")

    average_loss = sum_loss/len(data_loader)
    average_psnr = sum_psnr/len(data_loader)
    loss_list[0] = average_psnr
    if loss_list[0] > loss_list[1]:
        ms.save_checkpoint(network, save_path)
        print(f'Epoch{i+1}参数已保存,psnr:{loss_list[0]}')
        print(f'Epoch{i + 1},ThisPsnr:{loss_list[0]},LastPsnr:{loss_list[1]}')
        loss_list[1] = loss_list[0]




