import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size, data_path, val_ratio=0.2):
    # Data transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformations for validation/testing (no augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'),
                                             transform=train_transform)
    
    # Calculate the number of samples for validation
    val_size = int(len(train_val_dataset) * val_ratio)
    train_size = len(train_val_dataset) - val_size

    train_dataset, val_dataset = random_split(train_val_dataset,
                                              [train_size, val_size])
    test_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'test'),
                                        transform=test_transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, train_val_dataset.classes


def visualize_dataset(dataloader, classes, num_images=20, save_path=None):
    # Get a batch of images and labels from the dataloader
    inputs, labels = next(iter(dataloader))

    # Create a grid of images
    plt.figure(figsize=(15, 10))
    for i in range(min(num_images, len(inputs))):  # Ensure we don't exceed available images
        ax = plt.subplot(4, 5, i + 1)
        img = inputs[i].permute(1, 2, 0).numpy()  # Convert tensor to numpy array
        img = (img * np.array([0.229, 0.224, 0.225]) \
               + np.array([0.485, 0.456, 0.406]))  # Denormalize the image
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Convert to uint8 and clip values
        
        ax.imshow(img)
        ax.set_title(classes[labels[i].item()])
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()