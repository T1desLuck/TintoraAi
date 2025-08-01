import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
from pytorch_msssim import ssim
from src.model.tintora_ai import TintoraAI
from torch.amp import autocast, GradScaler
import yaml
from torch.utils.checkpoint import checkpoint


class TintoraDataset(Dataset):
    def __init__(self, data_path, pad_divisor=16):
        self.data_path = data_path
        self.pad_divisor = pad_divisor
        self.bw_path = os.path.join(data_path, "bw")
        self.color_path = os.path.join(data_path, "color")
        self.label_path = os.path.join(data_path, "labels")

        if not all(os.path.exists(p) for p in [self.bw_path, self.color_path, self.label_path]):
            raise FileNotFoundError(f"Одна из папок не найдена: {self.bw_path}, {self.color_path}, {self.label_path}")

        self.images = sorted([f for f in os.listdir(self.bw_path) if f.endswith((".jpg", ".png"))])
        self.labels = [f.replace(".jpg", ".npy").replace(".png", ".npy") for f in self.images]

        if not self.images:
            raise ValueError("Нет изображений в папке bw")

    def pad_image(self, image):
        w, h = image.size
        pad_w = (self.pad_divisor - w % self.pad_divisor) % self.pad_divisor
        pad_h = (self.pad_divisor - h % self.pad_divisor) % self.pad_divisor
        padded_image = Image.new(image.mode, (w + pad_w, h + pad_h), 0)
        padded_image.paste(image, (0, 0))
        return padded_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]
            bw_image = Image.open(os.path.join(self.bw_path, img_name)).convert("L")
            color_image = Image.open(os.path.join(self.color_path, img_name)).convert("RGB")
            padded_bw = self.pad_image(bw_image)
            padded_color = self.pad_image(color_image)
            bw_tensor = torch.from_numpy(np.array(padded_bw) / 255.0).float().unsqueeze(0)
            color_tensor = torch.from_numpy(np.array(padded_color) / 255.0).permute(2, 0, 1).float()
            label = np.load(os.path.join(self.label_path, self.labels[idx]))[0]
            return bw_tensor, color_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Ошибка загрузки {self.images[idx]}: {e}")
            return None


def filter_none_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def main():
    parser = argparse.ArgumentParser(description="Обучение TintoraAI")
    parser.add_argument("--config", type=str, default="config.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    model = TintoraAI(num_classes=config['num_classes']).to(device)
    criterion_color = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scaler = GradScaler('cuda')

    dataset = TintoraDataset(config['data_path'], pad_divisor=config['pad_divisor'])
    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=filter_none_collate)

    for epoch in range(config['epochs']):
        model.train()
        running_loss_color = 0.0
        running_loss_class = 0.0
        running_ssim = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, desc=f"Эпоха {epoch+1}/{config['epochs']}")):
            if batch is None:
                continue

            bw_images, color_images, labels = batch
            bw_images, color_images, labels = bw_images.to(device), color_images.to(device), labels.to(device)

            with autocast('cuda'):
                def model_forward(x):
                    return model(x)
                color_output, semantic_output = checkpoint(model_forward, bw_images)
                loss_color = criterion_color(color_output, color_images)
                loss_class = criterion_class(semantic_output, labels)
                loss = (loss_color + loss_class) / config['accum_steps']

            scaler.scale(loss).backward()
            if (i + 1) % config['accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss_color += loss_color.item()
            running_loss_class += loss_class.item()
            running_ssim += ssim(color_output, color_images, data_range=1.0, size_average=True).item()
            batch_count += 1

        if batch_count == 0:
            print("❌ Нет валидных батчей в эпохе")
            return

        avg_loss_color = running_loss_color / batch_count
        avg_loss_class = running_loss_class / batch_count
        avg_ssim = running_ssim / batch_count
        print(f"Эпоха {epoch+1}/{config['epochs']}, Color Loss: {avg_loss_color:.4f}, "
              f"Class Loss: {avg_loss_class:.4f}, SSIM: {avg_ssim:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"{config['save_path']}_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Чекпоинт сохранён: {checkpoint_path}")

    final_path = config['save_path']
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"Финальная модель сохранена: {final_path}")


if __name__ == "__main__":
    main()
