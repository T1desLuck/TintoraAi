import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """
    Класс для расчета GAN loss с поддержкой различных типов потерь
    """
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        if use_lsgan:
            # Least squares GAN loss (стабильнее)
            self.loss = nn.MSELoss()
        else:
            # Стандартный GAN loss с логистической функцией
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)


class PerceptualLoss(nn.Module):
    """
    Перцептивная потеря для улучшения визуального качества генерируемых изображений.
    Сравнивает характеристики изображений на разных уровнях абстракции.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Создаем свою сеть для извлечения признаков вместо VGG
        # Это позволит избежать использования предобученных весов
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Инициализируем веса случайным образом, но фиксируем их для стабильности
        for param in self.parameters():
            param.requires_grad = False
            
        # Слои нормализации для стабильности
        self.norm1 = nn.InstanceNorm2d(32)
        self.norm2 = nn.InstanceNorm2d(64)
        self.norm3 = nn.InstanceNorm2d(128)
        self.norm4 = nn.InstanceNorm2d(256)

    def forward(self, x, y):
        # Масштабируем входные изображения к единому размеру
        if x.shape[-2:] != y.shape[-2:]:
            if x.shape[-2] * x.shape[-1] > y.shape[-2] * y.shape[-1]:
                x = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)
            else:
                y = F.interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=True)
        
        # Извлекаем признаки разных уровней
        x1 = self.relu(self.norm1(self.conv1(x)))
        y1 = self.relu(self.norm1(self.conv1(y)))
        
        x2 = self.relu(self.norm2(self.conv2(x1)))
        y2 = self.relu(self.norm2(self.conv2(y1)))
        
        x3 = self.relu(self.norm3(self.conv3(x2)))
        y3 = self.relu(self.norm3(self.conv3(y2)))
        
        x4 = self.relu(self.norm4(self.conv4(x3)))
        y4 = self.relu(self.norm4(self.conv4(y3)))
        
        # Вычисляем потери на разных уровнях
        loss1 = F.mse_loss(x1, y1)
        loss2 = F.mse_loss(x2, y2)
        loss3 = F.mse_loss(x3, y3)
        loss4 = F.mse_loss(x4, y4)
        
        # Взвешенная сумма потерь
        # Придаем больший вес более высоким уровням для сохранения структурных элементов
        total_loss = 0.1 * loss1 + 0.2 * loss2 + 0.3 * loss3 + 0.4 * loss4
        
        return total_loss
