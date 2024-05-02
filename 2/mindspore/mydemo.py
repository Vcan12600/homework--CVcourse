import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, nn, Model
from mindspore.train.callback import LossMonitor
from mindspore.nn.metrics import Accuracy
from mindspore.common.initializer import TruncatedNormal
import mindspore.dataset.transforms as transforms
from mindspore.dataset import vision


# 设置设备为CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
mnist_dataset_dir = r"E:\mindspore\MNIST\raw"
label_trans = transforms.TypeCast(ms.int32)
data_trans = ms.dataset.transforms.Compose([transforms.TypeCast(ms.float32), vision.HWC2CHW()])
dataset_train = ds.MnistDataset(dataset_dir=mnist_dataset_dir, usage='train', shuffle=True).\
    map(operations=data_trans).batch(batch_size=4).map(operations=label_trans, input_columns=["label"])
dataset_test = ds.MnistDataset(dataset_dir=mnist_dataset_dir, usage='test', shuffle=True).\
    map(operations=data_trans).batch(batch_size=4).map(operations=label_trans, input_columns=["label"])


# 定义LeNet-5模型
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=6, kernel_size=5, padding=0, stride=1)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, stride=1)

        self.fc1 = nn.Dense(784, 120, weight_init=TruncatedNormal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=TruncatedNormal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=TruncatedNormal(0.02))

        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建LeNet-5模型实例
net = LeNet5()
save_path = r"E:\mindspore\LeNet5weight/LeNet5.ckpt"

# # 定义损失函数和优化器
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# optimizer = Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)
optimizer = ms.nn.Adam(params=net.trainable_params())

# 创建模型
model = Model(net, loss_fn, optimizer, metrics={'accuracy': Accuracy()})
# 训练模型
model.train(1, dataset_train, callbacks=[LossMonitor(1562)])
ms.save_checkpoint(net, save_path)
# 测试模型
result = model.eval(dataset_test)
print("Test Accuracy:", result['accuracy'])

