import sys 
import os 
sys.path.append(os.getcwd()) 

import torchvision.transforms as transforms 
import torch 

from Data_loader.Data_loader_test_mask import TestDataset
from Data_loader.Data_loader_train_mask import TrainDataset
from Data_loader.Data_loadertest_mask import NOTLFWestMaskDataset,LFWestMaskDataset
from config_mask import config

# Training data transforms
train_data_transforms = transforms.Compose([
    
    transforms.ToTensor(), # transform to tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# Testing data transform
test_data_transforms = transforms.Compose([
    transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])




train_dataloader = torch.utils.data.DataLoader(
    dataset = TrainDataset(
        face_dir = config['train_data_path'],
        mask_dir = config['mask_data_path'],
        csv_name = config['train_data_index'],
        num_triplets = config['num_train_triplets'],
        training_triplets_path = config['train_triplets_path'],
        transform = train_data_transforms,
        txt_mask='txt',
        predicter_path = config['predicter_path'],
        img_size = config['image_size']
    ),
    batch_size = config['train_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)


test_dataloader = torch.utils.data.DataLoader(
    dataset = TestDataset(
        dir = config['LFW_data_path'],
        pairs_path = config['LFW_pairs'],
        predicter_path = config['predicter_path'],
        img_size = config['image_size'],
        transform = test_data_transforms,
        test_pairs_paths = config['test_pairs_paths']
    ),
    batch_size = config['test_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)

# LFW masked data generate
LFWestMask_dataloader = torch.utils.data.DataLoader(
    dataset=LFWestMaskDataset(
        dir=config['LFW_data_path'],
        pairs_path=config['LFW_pairs'],
        predicter_path=config['predicter_path'],
        img_size=config['image_size'],
        transform=test_data_transforms,
        test_pairs_paths=config['test_pairs_paths']
    ),
    batch_size=config['test_batch_size'],
    num_workers=config['num_workers'],
    shuffle=False
)

# training data generate
from config_notmask import config as notcf
V9_train_dataloader = torch.utils.data.DataLoader(
    dataset = TrainDataset(
        face_dir = config['train_data_path'],
        mask_dir = notcf['mask_data_path'],
        csv_name = config['train_data_index'],
        num_triplets = config['num_train_triplets'],
        training_triplets_path = config['train_triplets_path'],
        transform = train_data_transforms,
        txt_mask='mask',
        predicter_path = config['predicter_path'],
        img_size = config['image_size']
    ),
    batch_size = config['train_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)

# LFW non masked data generate
NOTLFWestMask_dataloader = torch.utils.data.DataLoader(
    dataset = NOTLFWestMaskDataset(
        dir=config['LFW_data_path'],
        pairs_path=config['LFW_pairs'],
        predicter_path=config['predicter_path'],
        img_size=config['image_size'],
        transform=test_data_transforms,
        test_pairs_paths=config['test_pairs_paths']
    ),
    batch_size=config['test_batch_size'],
    num_workers=config['num_workers'],
    shuffle=False
)



