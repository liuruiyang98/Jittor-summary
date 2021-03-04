import jittor as jt
from jittor import init
from jittor import nn

# UNet++，插值增大尺寸后直接拼接特征，原论文
class NestedUNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2):
        super(NestedUNet, self).__init__()
        n1 = 64
        filters = [n1, (n1 * 2), (n1 * 4), (n1 * 8), (n1 * 16)]
        self.pool = nn.Pool(2, stride=2, op='maximum')
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv0_0 = DoubleConv(in_ch, filters[0], filters[0])
        self.conv1_0 = DoubleConv(filters[0], filters[1], filters[1])
        self.conv2_0 = DoubleConv(filters[1], filters[2], filters[2])
        self.conv3_0 = DoubleConv(filters[2], filters[3], filters[3])
        self.conv4_0 = DoubleConv(filters[3], filters[4], filters[4])
        self.conv0_1 = DoubleConv((filters[0] + filters[1]), filters[0], filters[0])
        self.conv1_1 = DoubleConv((filters[1] + filters[2]), filters[1], filters[1])
        self.conv2_1 = DoubleConv((filters[2] + filters[3]), filters[2], filters[2])
        self.conv3_1 = DoubleConv((filters[3] + filters[4]), filters[3], filters[3])
        self.conv0_2 = DoubleConv(((filters[0] * 2) + filters[1]), filters[0], filters[0])
        self.conv1_2 = DoubleConv(((filters[1] * 2) + filters[2]), filters[1], filters[1])
        self.conv2_2 = DoubleConv(((filters[2] * 2) + filters[3]), filters[2], filters[2])
        self.conv0_3 = DoubleConv(((filters[0] * 3) + filters[1]), filters[0], filters[0])
        self.conv1_3 = DoubleConv(((filters[1] * 3) + filters[2]), filters[1], filters[1])
        self.conv0_4 = DoubleConv(((filters[0] * 4) + filters[1]), filters[0], filters[0])
        self.final = nn.Conv(filters[0], n_classes, 1)

    def execute(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(jt.contrib.concat([x0_0, self.Up(x1_0)], dim=1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(jt.contrib.concat([x1_0, self.Up(x2_0)], dim=1))
        x0_2 = self.conv0_2(jt.contrib.concat([x0_0, x0_1, self.Up(x1_1)], dim=1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(jt.contrib.concat([x2_0, self.Up(x3_0)], dim=1))
        x1_2 = self.conv1_2(jt.contrib.concat([x1_0, x1_1, self.Up(x2_1)], dim=1))
        x0_3 = self.conv0_3(jt.contrib.concat([x0_0, x0_1, x0_2, self.Up(x1_2)], dim=1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(jt.contrib.concat([x3_0, self.Up(x4_0)], dim=1))
        x2_2 = self.conv2_2(jt.contrib.concat([x2_0, x2_1, self.Up(x3_1)], dim=1))
        x1_3 = self.conv1_3(jt.contrib.concat([x1_0, x1_1, x1_2, self.Up(x2_2)], dim=1))
        x0_4 = self.conv0_4(jt.contrib.concat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], dim=1))
        output = self.final(x0_4)
        return output

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv = DoubleConv(in_channels, out_channels, (in_channels // 2))
        else:
            self.up = nn.ConvTranspose(in_channels, (in_channels // 2), 2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def execute(self, x1, x2 = None):
        x = self.up(x1)
        if x2:
            x = jt.contrib.concat([x2, x], dim=1)
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch = None):
        super(DoubleConv, self).__init__()
        if (not mid_ch):
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv(in_ch, mid_ch, 3, padding=1), 
            nn.BatchNorm(mid_ch), 
            nn.ReLU(), 
            nn.Conv(mid_ch, out_ch, 3, padding=1), 
            nn.BatchNorm(out_ch), 
            nn.ReLU()
        )

    def execute(self, x):
        return self.double_conv(x)

