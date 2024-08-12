import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import re
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import pickle




class SegmentationDataset(Dataset):
    def __init__(self, image_files, depth_files, mask_files, transform=None):
        self.image_files = image_files
        self.depth_files = depth_files
        self.mask_files = mask_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        depth_path = self.depth_files[idx]
        mask_path = self.mask_files[idx]
        
        image = np.array(Image.open(img_path).convert("RGB"))  # 转换为RGB图像
        depth = np.array(Image.open(depth_path).convert("L"))  # 深度图仍为灰度图
        mask = np.array(Image.open(mask_path).convert("L"))    # mask为灰度图

        # 处理 mask，将50, 100, 150, 200对应的值映射到0, 1, 2, 3, 4五个类别
        mask = np.where(mask == 50, 1, mask)
        mask = np.where(mask == 100, 2, mask)
        mask = np.where(mask == 150, 3, mask)
        mask = np.where(mask == 200, 4, mask)
        mask = np.where(mask == 0, 0, mask)

        if self.transform:
            augmented = self.transform(image=image, depth=depth, mask=mask)
            image = augmented['image']
            depth = augmented['depth']
            mask = augmented['mask']

        # 将RGB图像和深度图像堆叠在一起作为输入
        # depth = depth.unsqueeze(0)  # 增加深度维度
        image = torch.cat((image, depth), dim=0)  # 合并RGB图和深度图 (4, 256, 256)

        return image, mask

def get_model():
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=4,                  # model input channels (3 for RGB + 1 for depth)
        classes=5,                      # model output channels (5 classes: background + 4 classes)
    )
    return model

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader):
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.long)  # CrossEntropyLoss expects long tensor
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    return model

def visualize_results(model, dataloader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(device, dtype=torch.float)
            outputs = model(images)
            outputs = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                rgb_image = images[i, :3, :, :].permute(1, 2, 0).cpu().numpy()  # 获取 RGB 图像
                
                ax[0].imshow(rgb_image)
                ax[0].set_title('Input Image')
                ax[1].imshow(masks[i].cpu().squeeze(), cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[2].imshow(outputs[i].cpu().squeeze(), cmap='gray')
                ax[2].set_title('Predicted Mask')
                plt.savefig(os.path.join(output_dir, f'result_{idx*images.size(0) + i}.png'))
                plt.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory containing the dataset with rgb, depth, and masks subdirectories")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--model_pth_dir_path", type=str, default="./", required=False, help="dir Path to read model_pth")
    parser.add_argument("--train", action='store_true', help="Whether to train the model")
    parser.add_argument("--test", action='store_true', help="Whether to test the model")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs to train")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the paths for the RGB, Depth, and Masks subdirectories
    rgb_path = os.path.join(args.dataset_path, 'cropped_RGB')
    depth_path = os.path.join(args.dataset_path, 'cropped_Depth')
    masks_path = os.path.join(args.dataset_path, 'cropped_mask')

    # Load file names and sort them
    image_files = sorted(
    [os.path.join(rgb_path, f) for f in os.listdir(rgb_path) if re.findall(r'\d+', f)],
    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
    )
    depth_files = sorted(
        [os.path.join(depth_path, f) for f in os.listdir(depth_path) if re.findall(r'\d+', f)],
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
    )
    mask_files = sorted(
        [os.path.join(masks_path, f) for f in os.listdir(masks_path) if re.findall(r'\d+', f)],
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
    )

    # Ensure output directories exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Split dataset
    split_file = os.path.join(args.dataset_path, 'data_split.pkl')
    if args.train:
        train_image_files, val_image_files, train_depth_files, val_depth_files, train_mask_files, val_mask_files = train_test_split(
            image_files, depth_files, mask_files, test_size=0.2, random_state=42
        )
        
        # Save the split to ensure the same split is used for training and testing
        split_data = {
            'train_image_files': train_image_files,
            'val_image_files': val_image_files,
            'train_depth_files': train_depth_files,
            'val_depth_files': val_depth_files,
            'train_mask_files': train_mask_files,
            'val_mask_files': val_mask_files
        }
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)
    else:
        # Load the split
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        
        train_image_files = split_data['train_image_files']
        val_image_files = split_data['val_image_files']
        train_depth_files = split_data['train_depth_files']
        val_depth_files = split_data['val_depth_files']
        train_mask_files = split_data['train_mask_files']
        val_mask_files = split_data['val_mask_files']

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)),  # 4通道输入
        ToTensorV2()
    ], additional_targets={'depth': 'image'})

    train_dataset = SegmentationDataset(train_image_files, train_depth_files, train_mask_files, transform=transform)
    val_dataset = SegmentationDataset(val_image_files, val_depth_files, val_mask_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = get_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if args.train:
        model = train(model, train_loader, criterion, optimizer, args.epochs, device)
        os.makedirs(args.model_pth_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.model_pth_dir_path, 'model.pth'))
        print('===========train done===========')
        
    if args.test:
        model.load_state_dict(torch.load(os.path.join(args.model_pth_dir_path, 'model.pth')))
        visualize_results(model, val_loader, device, os.path.join(args.output_dir, 'test'))
        print('===========test done============')

if __name__ == "__main__":
    main()
