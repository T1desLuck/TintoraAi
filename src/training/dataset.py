from torch.utils.data import Dataset
from PIL import Image
import glob
import os
import numpy as np
import torch


class ColorizationDataset(Dataset):
    def __init__(self, bw_path, color_path, label_path, transform=None):
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

    self.color_images = [p.replace("bw", "color") for p in self.bw_images]
    self.label_images = [p.replace("bw", label_path).replace(".jpg", ".npy") for p in self.bw_images]

    # Проверяем каждый файл
    for color_img in self.color_images:
        if not os.path.exists(color_img):
            raise FileNotFoundError(f"Цветное изображение отсутствует: {color_img}")
    for label_img in self.label_images:
        if not os.path.exists(label_img):
            raise FileNotFoundError(f"Метка отсутствует: {label_img}")

    self.transform = transform

    def __len__(self):
        return len(self.bw_images)

    def __getitem__(self, idx):
        bw_img = Image.open(self.bw_images[idx]).convert("L")
        color_img = Image.open(self.color_images[idx]).convert("RGB")
        label = np.load(self.label_images[idx])  # Метки как .npy
        if np.max(label) >= 1000:  # Проверка на превышение
            raise ValueError("Метки должны быть в диапазоне 0-999 для 1000 классов")
        if self.transform:
            bw_img = self.transform(bw_img)
            color_img = self.transform(color_img)
        label = torch.from_numpy(label).long()
        return bw_img, color_img, label
