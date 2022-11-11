"""
the implementation of ResNet50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False, groups=1):
        super(Bottleneck, self).__init__()
        stride = 2 if down_sample else 1
        mid_channels = out_channels // 4
        self.shortcut = Conv(in_channels, out_channels, kernel_size=1, stride=stride, activation=False) \
            if in_channels != out_channels else nn.Identity()
        self.conv = nn.Sequential(*[
            Conv(in_channels, mid_channels, kernel_size=1, stride=1),
            Conv(mid_channels, mid_channels, kernel_size=3, stride=stride, groups=groups),
            Conv(mid_channels, out_channels, kernel_size=1, stride=1, activation=False)
        ])

    def forward(self, x):
        y = self.conv(x) + self.shortcut(x)
        return F.relu(y, inplace=True)


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self._make_stage(64, 256, down_sample=False, num_blocks=3),
            self._make_stage(256, 512, down_sample=True, num_blocks=4),
            self._make_stage(512, 1024, down_sample=True, num_blocks=6),
            self._make_stage(1024, 2048, down_sample=True, num_blocks=3),
        ])
        self.head = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(2048, num_classes)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, down_sample, num_blocks):
        layers = [Bottleneck(in_channels, out_channels, down_sample=down_sample)]
        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels, out_channels, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


"""
resnet = ResNet50(num_classes=1000).train()
x = torch.randn(6, 3, 224, 224)
Out = resnet(x)
print(Out.shape)
"""

"""
class Bottleneck(nn.Module):
    # The multiple of the extension in each stage dimension
    extension = 4

    def __init__(self, in_channels, out_channels, stride, down_sample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.extension, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.extension)

        self.relu = nn.ReLU(inplace=True)

        # judge whether the residuals convolved
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.channels = 64
        super(ResNet50, self).__init__()

        # the parameter
        self.block = block
        self.layers = layers

        # the network layer of stem
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self.make_layer(self.block, 64, layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, layers[3], stride=2)

        # the following network
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.extension, num_classes)

    def forward(self, x):
        # stem: conv + bn + maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def make_layer(self, block, channels, block_num, stride=1):
        block_list = []
        down_sample = None
        if stride != 1 or self.channels != channels * block.extension:
            down_sample = nn.Sequential(
                nn.Conv2d(self.channels, channels * block.extension, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels * block.extension)
            )

        conv_block = block(self.channels, channels, stride=stride, down_sample=down_sample)
        block_list.append(conv_block)
        self.channels = channels * block.extension

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.channels, channels, stride=1))

        return nn.Sequential(*block_list)
"""
"""
resnet = ResNet50(Bottleneck, [3, 4, 6, 3], 1000).train()
x = torch.randn(64, 3, 224, 224)
Out = resnet(x)
print(Out.shape)
"""