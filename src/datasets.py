import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size, data_path, val_ratio=0.2):
    # Data transformations with augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize((224, 224)),
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