import matplotlib.pyplot as plt

def plot_loss(train_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()