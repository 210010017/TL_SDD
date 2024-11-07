import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SurfaceDefectDataset(Dataset):
    def __init__(self, root_dir, split='train', include_rare=True, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.include_rare = include_rare
        self.transform = transform
        self.image_paths = []
        self.labels = []

        defect_counts = {label: len(os.listdir(os.path.join(self.root_dir, self.split, str(label))))
                         for label in range(1, 11)}
        median_count = np.median(list(defect_counts.values()))
        self.general_classes = [k for k, v in defect_counts.items() if v >= median_count]
        self.rare_classes = [k for k, v in defect_counts.items() if v < median_count]

        selected_classes = self.general_classes if not self.include_rare else self.general_classes + self.rare_classes
        for label in selected_classes:
            folder_path = os.path.join(self.root_dir, self.split, str(label))
            for img_file in os.listdir(folder_path):
                self.image_paths.append(os.path.join(folder_path, img_file))
                self.labels.append(label - 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

def get_data_loader(root_dir, batch_size, split='train', include_rare=True):
    transform = train_transform if split == 'train' else test_transform
    dataset = SurfaceDefectDataset(root_dir=root_dir, split=split, include_rare=include_rare, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    return loader
