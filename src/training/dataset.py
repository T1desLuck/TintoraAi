from torch.utils.data import Dataset
from PIL import Image
import glob
from torchvision import transforms

class ColorizationDataset(Dataset):
    def __init__(self, bw_path, color_path, transform=None):
        self.bw_images = glob.glob(os.path.join(bw_path, "*.jpg"))
        self.color_images = [p.replace("bw", "color") for p in self.bw_images]
        self.transform = transform

    def __len__(self):
        return len(self.bw_images)

    def __getitem__(self, idx):
        bw_img = Image.open(self.bw_images[idx]).convert("L")
        color_img = Image.open(self.color_images[idx]).convert("RGB")
        if self.transform:
            bw_img = self.transform(bw_img)
            color_img = self.transform(color_img)
        return bw_img, color_img
