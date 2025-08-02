import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Двойная свертка с нормализацией и активацией"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """Блок внимания для улучшения качества деталей"""
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
        
        # Вычисляем матрицу внимания
        energy = torch.bmm(query.transpose(1, 2), key)
        attention = self.softmax(energy)
        
        # Применяем внимание к значениям
        out = torch.bmm(value, attention)
        out = out.view(batch, c, h, w)
        
        return x + self.gamma * out


class UNet(nn.Module):
    """U-Net архитектура с блоками внимания"""
    def __init__(self, in_channels=1, out_channels=3, dynamic_size=True):
        super(UNet, self).__init__()
        
        # Флаг для динамического размера (для различных входных размеров)
        self.dynamic_size = dynamic_size
        
        # Энкодер
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Бутылочное горлышко с вниманием
        self.bottleneck = DoubleConv(256, 512)
        self.attention = AttentionBlock(512)
        
        # Декодер
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
        # Сохраняем исходный размер для динамического изменения
        original_size = x.shape[-2:]
        
        # Проверяем минимальный размер
        min_size = 16  # Минимальный размер для 4 уровней свертки
        current_size = min(original_size)
        if current_size < min_size and self.dynamic_size:
            scale_factor = min_size / current_size
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        # Энкодер
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Бутылочное горлышко
        b = self.bottleneck(p4)
        b = self.attention(b)
        
        # Декодер с конкатенацией
        d4 = self.upconv4(b)
        # Проверяем соответствие размеров перед конкатенацией
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, e4.shape[-2:], mode='bilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, e3.shape[-2:], mode='bilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, e2.shape[-2:], mode='bilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, e1.shape[-2:], mode='bilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.out(d1)
        
        # Если размер изменился, возвращаем к исходному
        if self.dynamic_size and out.shape[-2:] != original_size:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)
            
        return torch.sigmoid(out)


class ObjectClassifier(nn.Module):
    """Классификатор для семантического анализа изображения"""
    def __init__(self, num_classes=100):
        super(ObjectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Адаптивный пул для любого размера входа
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Проверяем минимальный размер
        if min(x.shape[2], x.shape[3]) < 32:
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
            
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.adaptive_pool(x)  # Адаптивный пул обеспечивает фиксированный размер
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Generator(nn.Module):
    """Генератор для GAN, основан на модифицированной U-Net архитектуре"""
    def __init__(self, in_channels=1, out_channels=3):
        super(Generator, self).__init__()
        # Используем модифицированный U-Net как основу генератора
        self.unet = UNet(in_channels, out_channels, dynamic_size=True)
        
        # Добавляем финальные слои для уточнения деталей
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Генерируем базовое цветное изображение
        base_output = self.unet(x)
        # Уточняем детали
        refined_output = self.refine(base_output)
        return refined_output


class Discriminator(nn.Module):
    """PatchGAN дискриминатор для GAN"""
    def __init__(self, in_channels=4):  # 1 (ЧБ) + 3 (цветное) каналы
        super(Discriminator, self).__init__()
        
        # PatchGAN дискриминатор (анализирует патчи изображения)
        def discriminator_block(in_f, out_f, normalization=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )
        
    def forward(self, img_bw, img_color):
        # Объединяем ЧБ и цветное изображения
        # Если размеры разные, приводим к одному размеру
        if img_bw.shape[-2:] != img_color.shape[-2:]:
            img_bw = F.interpolate(img_bw, size=img_color.shape[-2:], mode='bilinear', align_corners=True)
            
        img_input = torch.cat([img_bw, img_color], 1)
        return self.model(img_input)


class TintoraAI(nn.Module):
    """Основная модель TintoraAI, объединяющая Generator (U-Net), GAN и классификатор"""
    def __init__(self, num_classes=100):
        super(TintoraAI, self).__init__()
        # Генератор (модель колоризации)
        self.generator = Generator(in_channels=1, out_channels=3)
        # Дискриминатор для GAN
        self.discriminator = Discriminator(in_channels=4)
        # Классификатор для семантического анализа
        self.classifier = ObjectClassifier(num_classes)
        
        # Флаг для переключения между режимами обучения и инференса
        self.training_gan = False

    def forward(self, x):
        # Генерируем цветное изображение
        color_output = self.generator(x)
        
        # Получаем семантическую информацию
        semantic_output = self.classifier(x)
        
        # В режиме обучения GAN возвращаем также выход дискриминатора
        if self.training_gan:
            disc_output = self.discriminator(x, color_output)
            return color_output, semantic_output, disc_output
        
        return color_output, semantic_output
