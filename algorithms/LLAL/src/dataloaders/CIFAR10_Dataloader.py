import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class CIFAR10Dataloader(Dataset):
    def __init__(self, src_path, sample_paths, class_labels):
        self.image_transformer = transforms.Compose([transforms.RandomHorizontalFlip(), 
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(), 
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        self.src_path = src_path
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]

        return sample, class_label


class CIFAR10_Test_Dataloader(CIFAR10Dataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])