from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import numpy as np
import torch
import random
from torchvision import transforms


class ColorizationDataset(Dataset):
    """Датасет для обучения колоризации изображений с поддержкой различных размеров"""
    def __init__(self, bw_path, color_path, label_path, 
                 transform=None, augment=True, min_size=64, max_size=1024):
        # Проверяем, что папки существуют
        if not os.path.exists(bw_path):
            raise FileNotFoundError(f"Папка с ЧБ изображениями не найдена: {bw_path}")
        if not os.path.exists(color_path):
            raise FileNotFoundError(f"Папка с цветными изображениями не найдена: {color_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Папка с метками не найдена: {label_path}")

        # Ищем файлы (поддержка .jpg и .png)
        self.bw_images = sorted(glob.glob(os.path.join(bw_path, "*.[jp][pn]g")))
        if not self.bw_images:
            raise ValueError(f"В папке {bw_path} нет изображений .jpg или .png")

        # Формируем пути к соответствующим цветным изображениям и меткам
        self.color_images = [p.replace(bw_path, color_path) for p in self.bw_images]
        self.label_files = [p.replace(bw_path, label_path).replace(".jpg", ".npy").replace(".png", ".npy") 
                            for p in self.bw_images]

        # Проверяем каждый файл
        for color_img in self.color_images:
            if not os.path.exists(color_img):
                raise FileNotFoundError(f"Цветное изображение отсутствует: {color_img}")
        for label_file in self.label_files:
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Метка отсутствует: {label_file}")

        self.transform = transform
        self.augment = augment
        self.min_size = min_size
        self.max_size = max_size
        
        # Базовые преобразования для аугментации
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
        ])

    def __len__(self):
        return len(self.bw_images)

    def __getitem__(self, idx):
        # Загружаем изображения
        bw_img = Image.open(self.bw_images[idx]).convert("L")
        color_img = Image.open(self.color_images[idx]).convert("RGB")
        
        # Проверяем размеры
        w, h = bw_img.size
        
        # Ограничиваем максимальный размер для экономии памяти
        if w > self.max_size or h > self.max_size:
            scale = min(self.max_size / w, self.max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            bw_img = bw_img.resize((new_w, new_h), Image.LANCZOS)
            color_img = color_img.resize((new_w, new_h), Image.LANCZOS)
            
        # Если изображение слишком маленькое, увеличиваем
        if w < self.min_size or h < self.min_size:
            scale = max(self.min_size / w, self.min_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            bw_img = bw_img.resize((new_w, new_h), Image.LANCZOS)
            color_img = color_img.resize((new_w, new_h), Image.LANCZOS)
            
        # Убеждаемся, что размеры совпадают
        if color_img.size != bw_img.size:
            color_img = color_img.resize(bw_img.size, Image.LANCZOS)
            
        # Аугментация данных
        if self.augment and random.random() > 0.5:
            # Применяем одинаковые преобразования к обоим изображениям
            seed = torch.randint(0, 2147483647, (1,))[0].item()
            random.seed(seed)
            torch.manual_seed(seed)
            bw_img = self.augmentation(bw_img)
            
            random.seed(seed)
            torch.manual_seed(seed)
            color_img = self.augmentation(color_img)
            
        # Загружаем метки
        label = np.load(self.label_files[idx])
        if np.max(label) >= 100:  # Проверка на превышение
            raise ValueError(f"Метки должны быть в диапазоне 0-99 для 100 классов, но найдено {np.max(label)}")
            
        # Применяем дополнительные трансформации
        if self.transform:
            bw_img = self.transform(bw_img)
            color_img = self.transform(color_img)
            
        # Преобразуем в тензоры
        bw_tensor = transforms.ToTensor()(bw_img)
        color_tensor = transforms.ToTensor()(color_img)
        label_tensor = torch.from_numpy(label).long()
        
        return bw_tensor, color_tensor, label_tensor
