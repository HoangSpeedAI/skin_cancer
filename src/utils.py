from datetime import datetime
import os
import matplotlib.pyplot as plt
import torch

def plot_learning_rates(lrs, session_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(lrs, label='Learning Rate', color='blue')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(session_dir, 'learning_rate_schedule.png'))
    plt.close()


def plot_losses(train_losses, val_losses, session_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(session_dir, 'loss_plot.png'))
    plt.close()


def create_session_directory(base_dir, architecture, config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(base_dir, f"{architecture}_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)

    with open(os.path.join(session_dir, 'config.txt'), 'w') as config_file:
        config_file.write(str(config))

    return session_dir

def create_scheduler(optimizer, config):
    """
    The function `create_scheduler` returns a learning rate scheduler based on the specified
    configuration.
    
    :param optimizer: The `optimizer` parameter is an instance of the optimizer class in PyTorch, such
    as `torch.optim.SGD` or `torch.optim.Adam`, that is responsible for updating the weights of the
    neural network during training based on the computed gradients
    :param config: config is a dictionary containing configuration parameters for the scheduler. It
    includes the following keys:
    :return: The function `create_scheduler` returns a learning rate scheduler based on the
    configuration provided. It returns one of the following types of schedulers based on the value of
    `config['scheduler']`:
    1. If `config['scheduler']` is 'step', it returns a StepLR scheduler.
    2. If `config['scheduler']` is 'cyclic', it returns a CyclicLR scheduler
    """
    if config['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif config['scheduler'] == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                 base_lr=1e-5,
                                                 max_lr=optimizer.param_groups[0]['lr'],
                                                 step_size_up=config['step_size_up'],
                                                 mode='triangular')
    elif config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                          T_max=config['num_epochs'])
    else:
        print(f"Unknown scheduler: {config['scheduler']}")
        return None
