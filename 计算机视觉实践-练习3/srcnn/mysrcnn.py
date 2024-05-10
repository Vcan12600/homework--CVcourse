from mindspore import nn


# 定义SrCnn模型
class SrCnn(nn.Cell):
    def __init__(self):
        super(SrCnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, padding=0,
                               stride=1, pad_mode="pad", has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0,
                               stride=1, pad_mode="pad", has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=0,
                               stride=1, pad_mode="pad", has_bias=True)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.norm3 = nn.BatchNorm2d(num_features=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return x


class SrCnnTest(nn.Cell):
    def __init__(self):
        super(SrCnnTest, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9,
                               stride=1, has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1,
                               stride=1, has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5,
                               stride=1, has_bias=True)
        self.norm1 = nn.BatchNorm2d(num_features=64)
        self.norm2 = nn.BatchNorm2d(num_features=32)
        self.norm3 = nn.BatchNorm2d(num_features=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return x
