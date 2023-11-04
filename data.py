import torch
import torchvision
from torchvision.datasets import Food101, MNIST
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

split = 0.25
batch_size = 32

train_datasets = Food101(root = "./data", split = "train", transform = transform, download = True)
test_datasets = Food101(root = "./data", split = "test", transform = transform, download = True)

num_samples_train = int(split * len(train_datasets))
train_indices = torch.randperm(len(train_datasets))[:num_samples_train]

num_samples_valid = int(split * len(train_datasets))
valid_indices = torch.randperm(len(train_datasets))[:num_samples_valid]

num_samples_test = int(split * len(test_datasets))
test_indices = torch.randperm(len(test_datasets))[:num_samples_test]

split_tarin_data = Subset(train_datasets, train_indices)
split_valid_data = Subset(train_datasets, valid_indices)
split_test_data = Subset(test_datasets, test_indices)

train_dataloader = DataLoader(split_tarin_data, batch_size = batch_size, shuffle = True, drop_last = True)
valid_dataloader = DataLoader(split_valid_data, batch_size = batch_size, shuffle = True, drop_last = True)
test_dataloader = DataLoader(split_test_data, batch_size = batch_size, shuffle = False, drop_last = True)