import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from src.model.tintora_ai import TintoraAI
from src.training.dataset import ColorizationDataset
import argparse
import os


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def train_model(
    data_path, epochs=10, batch_size=8,
    save_path="colorizer_weights.pth", num_classes=99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ColorizationDataset(
        bw_path=os.path.join(data_path, "bw"),
        color_path=os.path.join(data_path, "color"),
        label_path=os.path.join(data_path, "labels"),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    generator = TintoraAI(num_classes=num_classes).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    criterion_color = nn.MSELoss()
    criterion_semantic = nn.CrossEntropyLoss()
    criterion_gan = nn.BCELoss()

    def perceptual_loss(output, target):
        vgg_layers = [2, 7, 12]
        loss = 0
        for layer in vgg_layers:
            vgg_output = vgg[:layer](output)
            vgg_target = vgg[:layer](target)
            loss += F.mse_loss(vgg_output, vgg_target)
        return loss / len(vgg_layers)

    for epoch in range(epochs):
        for bw_imgs, color_imgs, labels in dataloader:
            bw_imgs, color_imgs, labels = (
                bw_imgs.to(device),
                color_imgs.to(device),
                labels.to(device)
            )

            # Train Discriminator
            optimizer_d.zero_grad()
            real_validity = discriminator(color_imgs)
            fake_imgs, _ = generator(bw_imgs)
            fake_validity = discriminator(fake_imgs.detach())
            d_loss = (
                criterion_gan(real_validity, torch.ones_like(real_validity)) +
                criterion_gan(fake_validity, torch.zeros_like(fake_validity))
            ) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_imgs, semantic_output = generator(bw_imgs)
            fake_validity = discriminator(fake_imgs)
            pixel_loss = criterion_color(fake_imgs, color_imgs)
            perc_loss = perceptual_loss(fake_imgs, color_imgs)
            g_color_loss = 0.7 * pixel_loss + 0.3 * perc_loss
            g_semantic_loss = criterion_semantic(semantic_output, labels)
            g_gan_loss = criterion_gan(fake_validity, torch.ones_like(fake_validity))
            total_loss = (
                g_color_loss +
                0.1 * g_semantic_loss +
                0.1 * g_gan_loss
            )
            total_loss.backward()
            optimizer_g.step()

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}"
        )

    torch.save(generator.state_dict(), save_path)
    return generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TintoraAI model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_classes", type=int, default=99,
                        help="Number of classes for semantic output")
    args = parser.parse_args()
    train_model(args.data_path, args.epochs, 
                args.batch_size, num_classes=args.num_classes)
