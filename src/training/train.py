import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
from src.model.tintora_ai import TintoraAI
from src.training.dataset import ColorizationDataset
import argparse
import os


def train_model(data_path, epochs=10, batch_size=8, save_path="colorizer_weights.pth"):
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

    model = TintoraAI().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Perceptual loss с VGG16
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    criterion_color = nn.MSELoss()
    criterion_semantic = nn.CrossEntropyLoss()

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
            bw_imgs, color_imgs, labels = bw_imgs.to(device), color_imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            color_output, semantic_output = model(bw_imgs)
            
            # Color loss
            pixel_loss = criterion_color(color_output, color_imgs)
            perc_loss = perceptual_loss(color_output, color_imgs)
            total_color_loss = 0.7 * pixel_loss + 0.3 * perc_loss

            # Semantic loss
            semantic_loss = criterion_semantic(semantic_output, labels)

            # Total loss
            total_loss = total_color_loss + 0.1 * semantic_loss
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item()}")

    torch.save(model.state_dict(), save_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TintoraAI model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    args = parser.parse_args()
    train_model(args.data_path, args.epochs, args.batch_size)
