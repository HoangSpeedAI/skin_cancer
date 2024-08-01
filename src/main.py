import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import get_dataloaders
from model import get_model
from train import train
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a PyTorch model.')
    parser.add_argument('--config', type=str, default='configs/resnet18_config.yaml',
                        help='Path to the config file')
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

    # Get data loaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size,
                                                                     data_path)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(input_channels, output_channels, architecture).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # For binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Evaluate the model
    evaluate(model, test_loader, device)