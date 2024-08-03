import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import get_dataloaders, visualize_dataset
from model import get_model
from train import train
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train or test a PyTorch model.')
    parser.add_argument('--config', '-c', type=str,
                        default='configs/resnet18_config.yaml',
                        help='Path to the config file')
    parser.add_argument('--mode', '-m', type=str, choices=['train', 'test'], required=True,
                        help='Mode to run: train or test')
    parser.add_argument('--model-path', '-mp', type=str,
                        help='Path to the model file for testing (required in test mode)')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)

    # Hyperparameters from config
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    data_path = config['data_path']
    input_channels = config['input_channels']
    output_channels = config['output_channels']
    architecture = config['architecture']

    base_dir = 'experiments'

    # Get data loaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size, data_path)
    
    # Visualize dataset samples
    visualize_dataset(train_loader, classes, save_path='viz_train.png')
    visualize_dataset(val_loader, classes, save_path='viz_val.png')

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(input_channels, output_channels, architecture).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4   # Set weight decay (L2 regularization)
    )

    if args.mode == 'train':
        # Train the model
        best_model_path = train(model, train_loader, val_loader, criterion, optimizer,
                                num_epochs, device, architecture, base_dir, config)

    elif args.mode == 'test':
        # Ensure model path is provided
        if not args.model_path:
            raise ValueError("Model path must be specified in test mode.")
        
        test(model, args.model_path, test_loader, device)