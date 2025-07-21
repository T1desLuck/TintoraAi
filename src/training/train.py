import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from src.model.tintora_ai import TintoraAI
from src.training.dataset import ColorizationDataset
import argparse


def train_model(data_path, epochs=10, batch_size=8, save_path="colorizer_weights.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ColorizationDataset(
        bw_path=os.path.join(data_path, "bw"),
        color_path=os.path.join(data_path, "color"),
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TintoraAI().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Perceptual loss с VGG16
    vgg = models.vgg16(pretrained=True).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    def perceptual_loss(output, target):
        vgg_layers = [2, 7, 12]
        loss = 0
        for layer in vgg_layers:
            vgg_output = vgg[:layer](output)
            vgg_target = vgg[:layer](target)
            loss += F.mse_loss(vgg_output, vgg_target)
        return loss / len(vgg_layers)

    criterion = nn.MSELoss()

    for epoch in range(epochs):
        for bw_imgs, color_imgs in dataloader:
            bw_imgs, color_imgs = bw_imgs.to(device), color_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(bw_imgs)
            pixel_loss = criterion(outputs, color_imgs)
            perc_loss = perceptual_loss(outputs, color_imgs)
            loss = 0.7 * pixel_loss + 0.3 * perc_loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), save_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TintoraAI model")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    args = parser.parse_args()
    train_model(args.data_path, args.epochs, args.batch_size)
