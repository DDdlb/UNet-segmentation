import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        layer = []

        layer.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=3, padding=1))
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.ReLU(inplace=True))
        layer.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        layer.append(nn.BatchNorm2d(out_channels))
        layer.append(nn.ReLU(inplace=True))

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, in_channels2, out_channels):
        super(UpSample, self).__init__()
        self.upSample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels+in_channels2, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.upSample(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.downSample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = ConvBlock(in_channels=3, out_channels=64)
        self.layer2 = ConvBlock(in_channels=64, out_channels=128)
        self.layer3 = ConvBlock(in_channels=128, out_channels=256)
        self.layer4 = ConvBlock(in_channels=256, out_channels=512)
        self.layer5 = ConvBlock(in_channels=512, out_channels=1024)
        self.up1 = UpSample(in_channels=1024, in_channels2=512, out_channels=512)
        self.up2 = UpSample(in_channels=512, in_channels2=256, out_channels=256)
        self.up3 = UpSample(in_channels=256, in_channels2=128, out_channels=128)
        self.up4 = UpSample(in_channels=128, in_channels2=64, out_channels=64)
        self.down_channel = nn.Sequential(

                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

        )


    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(self.downSample(layer1))
        layer3 = self.layer3(self.downSample(layer2))
        layer4 = self.layer4(self.downSample(layer3))
        layer5 = self.layer5(self.downSample(layer4))

        out = self.up1(layer5, layer4)
        out = self.up2(out, layer3)
        out = self.up3(out, layer2)
        out = self.up4(out, layer1)

        out = self.down_channel(out)

        return out



if __name__ == '__main__':
    x = torch.randn(1, 3, 256, 256)
    net = UNet()
    print(net(x).shape)

