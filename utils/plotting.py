import matplotlib.pyplot as plt

def plot_loss(losses):
    fig, ax = plt.subplots()
    ax.plot(losses, marker='o', linestyle='-', color='blue')
    ax.set_title("Training Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    plt.tight_layout()
    return fig

def plot_accuracy(accuracies):
    fig, ax = plt.subplots()
    ax.plot(accuracies, marker='o', linestyle='-', color='green')
    ax.set_title("Test Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(True)
    plt.tight_layout()
    return fig
