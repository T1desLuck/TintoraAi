import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, c, h, w = x.size()
        if h * w == 0:
            return x
        query = self.query_conv(x).view(batch, -1, h * w)
        key = self.key_conv(x).view(batch, -1, h * w)
        value = self.value_conv(x).view(batch, c, h * w)
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = self.softmax(energy)
        out = torch.bmm(value, attention)
        out = out.view(batch, c, h, w)
        return x + self.gamma * out

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(256, 512)
        self.attention = AttentionBlock(512)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        b = self.bottleneck(p4)
        b = self.attention(b)
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out(d1)
        return torch.sigmoid(out)

class ObjectClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(ObjectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.shape[2] < 64 or x.shape[3] < 64:
            raise ValueError("Входное изображение слишком маленькое. Минимальный размер 64x64")
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class TintoraAI(nn.Module):
    def __init__(self, num_classes=100):
        super(TintoraAI, self).__init__()
        self.unet = UNet()
        self.classifier = ObjectClassifier(num_classes)

    def forward(self, x):
        color_output = self.unet(x)
        semantic_output = self.classifier(x)
        return color_output, semantic_output
