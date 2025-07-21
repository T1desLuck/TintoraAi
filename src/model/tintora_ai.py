import torch
import torch.nn as nn
import torch.nn.functional as F


class TintoraAI(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(TintoraAI, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='reflect'),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def attention_block(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 1),
                nn.Sigmoid()
            )

        # Энкодер (легкий, для мобильности)
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)

        # Бутылочное горлышко с attention
        self.bottleneck = conv_block(128, 256)
        self.attention = attention_block(256)

        # Декодер
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Энкодер
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        # Attention
        attn = self.attention(b)
        b = b * attn

        # Декодер
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))
