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
from src.model.preprocess import preprocess_image
from torch.amp import autocast, GradScaler
import sys


class TintoraDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.bw_path = os.path.join(data_path, "bw")
        self.color_path = os.path.join(data_path, "color")
        self.label_path = os.path.join(data_path, "labels")
        if not all(os.path.exists(p) for p in [self.bw_path, self.color_path, self.label_path]):
            raise FileNotFoundError(f"Одна из папок не найдена: {self.bw_path}, {self.color_path}, {self.label_path}")
        self.images = sorted([f for f in os.listdir(self.bw_path) if f.endswith((".jpg", ".png"))])
        self.labels = [f.replace(".jpg", ".npy").replace(".png", ".npy") for f in self.images]
        if not self.images:
            raise ValueError("Нет изображений в папке bw")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]
            bw_image = Image.open(os.path.join(self.bw_path, img_name)).convert("L")
            color_image = Image.open(os.path.join(self.color_path, img_name)).convert("RGB")
            label = np.load(os.path.join(self.label_path, self.labels[idx]))[0]
            if bw_image.size != (512, 512) or color_image.size != (512, 512):
                raise ValueError(f"Image {img_name} size mismatch, expected 512x512")
            bw_tensor, _ = preprocess_image(bw_image, min_size=512)
            color_tensor = torch.from_numpy(np.array(color_image) / 255.0).permute(2, 0, 1).float()
            return bw_tensor, color_tensor, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Train TintoraAI")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset (bw/, color/, labels/)")
    parser.add_argument("--save_path", type=str, default="models/colorizer_weights.pth", help="Path to save weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TintoraAI(num_classes=100).to(device)
    criterion_color = nn.MSELoss()
    criterion_class = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler('cuda')
    print("LPIPS отключён для исключения предобученных весов")

    dataset = TintoraDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss_color = 0.0
        running_loss_class = 0.0
        running_ssim = 0.0
        batch_count = 0
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            if batch is None:
                continue
            bw_images, color_images, labels = batch
            if bw_images is None or color_images is None or labels is None:
                continue
            bw_images, color_images, labels = bw_images.to(device), color_images.to(device), labels.to(device)

            print(f"Batch tensor size: {bw_images.shape}")
            with autocast('cuda'):
                color_output, semantic_output = model(bw_images)
                loss_color = criterion_color(color_output, color_images)
                loss_class = criterion_class(semantic_output, labels)
                loss = loss_color + loss_class

            loss = loss / args.accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss_color += loss_color.item()
            running_loss_class += loss_class.item()
            ssim_val = ssim(color_output, color_images, data_range=1.0, size_average=True)
            running_ssim += ssim_val.item()
            batch_count += 1

        if batch_count == 0:
            print("❌ Нет валидных батчей в эпохе")
            sys.exit(1)

        avg_loss_color = running_loss_color / batch_count
        avg_loss_class = running_loss_class / batch_count
        avg_ssim = running_ssim / batch_count
        print(f"Epoch {epoch+1}/{args.epochs}, Color Loss: {avg_loss_color:.4f}, "
              f"Class Loss: {avg_loss_class:.4f}, SSIM: {avg_ssim:.4f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"models/checkpoint_epoch_{epoch+1}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'color_loss': avg_loss_color,
                'class_loss': avg_loss_class,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(model.state_dict(), args.save_path)
    print(f"Final model saved to {args.save_path}")


if __name__ == "__main__":
    main()
