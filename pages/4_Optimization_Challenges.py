import os
os.environ['STREAMLIT_SERVER_WATCH_EXCLUSIONS'] = 'torch'

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import io
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(layout="wide", page_title="Optimization Challenges")

# Title and introduction
st.title("Optimization Challenges in Machine Learning")
st.markdown("""
This app demonstrates how different optimization algorithms perform on real machine learning tasks.
You can compare various optimizers on two different tasks:

1. **Binary Classification**: Simple MLP on a synthetic moon-shaped dataset
2. **Image Classification**: ResNet18 on CIFAR-10 dataset

Observe how different optimizers affect learning speed, stability, and final performance!
""")

# -------------------------------------------------------------------
# Models & Dataset Definitions
# -------------------------------------------------------------------

# 1. Simple MLP for the moon dataset
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# 2. ResNet18 for CIFAR-10
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(512, 10)
        self.model = base

    def forward(self, x):
        return self.model(x)

# Function to prepare the moon dataset
@st.cache_data
def prepare_moon_dataset(n_samples=1000, noise=0.2, test_size=0.2, random_state=42, batch_size=32):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    
    return X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader

# Function to prepare the CIFAR-10 dataset
@st.cache_resource
def prepare_cifar10_dataset(subset_percentage=0.2, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Take a subset of training data
    train_len = int(len(full_train) * subset_percentage)
    subset_train, _ = random_split(full_train, [train_len, len(full_train) - train_len])
    
    trainloader = DataLoader(subset_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader, full_train.classes

# Training functions
def train_moon_model(optimizer_name, train_loader, X_test_tensor, y_test_tensor, epochs=50, learning_rate=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = SimpleMLP().to(device)
    criterion = nn.BCELoss()
    
    # Set optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'NAG':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate/10)  # RMSProp typically needs lower LR
    elif optimizer_name == 'AdaDelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)  # AdaDelta has its own adaptive LR
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)  # Adam typically needs lower LR
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Move data to device
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    # Training history
    loss_history = []
    acc_history = []
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            test_loss = criterion(outputs, y_test_tensor).item()
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_tensor).float().mean().item()
            
        loss_history.append(test_loss)
        acc_history.append(accuracy)
        
        # For debugging
        if epoch % 10 == 0:
            print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs}, Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
            
    return model, loss_history, acc_history

def train_cifar_model(optimizer_name, trainloader, testloader, epochs=10, learning_rate=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'NAG':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name == 'AdaGrad':
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdaDelta':
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    # Training history
    loss_history = []
    acc_history = []
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for xb, yb in trainloader:
            xb, yb = xb.to(device), yb.to(device)
            
            # Forward pass
            outputs = model(xb)
            loss = criterion(outputs, yb)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(trainloader)
        loss_history.append(epoch_loss)
        
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for xb, yb in testloader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
                
        accuracy = correct / total
        acc_history.append(accuracy)
        
        # For debugging
        print(f"[{optimizer_name}] Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    return model, loss_history, acc_history

# Visualization functions
def plot_moon_dataset(X_train, X_test, y_train, y_test):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training set
    ax[0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], 
                color='steelblue', label='Class 0', alpha=0.7)
    ax[0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], 
                color='darkorange', label='Class 1', alpha=0.7)
    ax[0].set_title("Training Dataset")
    ax[0].set_xlabel("Feature 1")
    ax[0].set_ylabel("Feature 2")
    ax[0].legend()
    ax[0].grid(True)
    
    # Plot test set
    ax[1].scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], 
                color='steelblue', label='Class 0', alpha=0.7)
    ax[1].scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], 
                color='darkorange', label='Class 1', alpha=0.7)
    ax[1].set_title("Test Dataset")
    ax[1].set_xlabel("Feature 1")
    ax[1].set_ylabel("Feature 2")
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    return fig

def plot_decision_boundary(model, X_test_tensor, y_test_tensor, device='cpu'):
    model.eval()
    
    # Create a grid to visualize decision boundary
    h = 0.01
    x_min, x_max = X_test_tensor[:, 0].min() - 1, X_test_tensor[:, 0].max() + 1
    y_min, y_max = X_test_tensor[:, 1].min() - 1, X_test_tensor[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Make predictions for every point in the grid
    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    with torch.no_grad():
        Z = model(grid_tensor).cpu().numpy()
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary and test points
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    
    # Plot test points
    X_test_np = X_test_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy().ravel()
    ax.scatter(X_test_np[y_test_np == 0][:, 0], X_test_np[y_test_np == 0][:, 1], 
              color='steelblue', label='Class 0', edgecolors='k')
    ax.scatter(X_test_np[y_test_np == 1][:, 0], X_test_np[y_test_np == 1][:, 1], 
              color='darkorange', label='Class 1', edgecolors='k')
    
    ax.set_title("Decision Boundary and Test Data")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True)
    
    return fig

def plot_cifar_examples(trainloader, classes, num_examples=10):
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # Show images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_examples):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        img = img / 2 + 0.5  # Unnormalize
        axes[i].imshow(img)
        axes[i].set_title(f"{classes[labels[i]]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_curves(results, title="Training Results"):
    """Plot loss and accuracy curves for multiple optimizers"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves
    for opt_name, (loss_hist, _) in results.items():
        ax1.plot(loss_hist, label=opt_name)
    
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves
    for opt_name, (_, acc_hist) in results.items():
        ax2.plot(acc_hist, label=opt_name)
    
    ax2.set_title(f"{title} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# Animated training progress
def create_animated_curves(results, epoch, title="Training Progress"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot loss curves up to current epoch
    for opt_name, (loss_hist, _) in results.items():
        current_loss = loss_hist[:epoch+1]
        ax1.plot(current_loss, label=opt_name)
    
    ax1.set_title(f"{title} - Loss (Epoch {epoch+1})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy curves up to current epoch
    for opt_name, (_, acc_hist) in results.items():
        current_acc = acc_hist[:epoch+1]
        ax2.plot(current_acc, label=opt_name)
    
    ax2.set_title(f"{title} - Accuracy (Epoch {epoch+1})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# Main App Logic
# -------------------------------------------------------------------

def main():
    # Sidebar for task selection and parameters
    st.sidebar.header("Task Selection")
    task = st.sidebar.radio("Select Task", [
        "Binary Classification (Moon Dataset)", 
        "Image Classification (CIFAR-10)"
    ])
    
    # Common optimizer options
    optimizer_options = ['SGD', 'Momentum', 'NAG', 'AdaGrad', 'RMSProp', 'AdaDelta', 'Adam']
    
    if task == "Binary Classification (Moon Dataset)":
        # Show dataset parameters
        st.sidebar.subheader("Dataset Parameters")
        n_samples = st.sidebar.slider("Number of samples", 500, 5000, 1000)
        noise = st.sidebar.slider("Dataset noise", 0.0, 0.5, 0.2, step=0.05)
        
        # Show training parameters
        st.sidebar.subheader("Training Parameters")
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, step=0.001)
        epochs = st.sidebar.slider("Number of Epochs", 10, 100, 50)
        batch_size = st.sidebar.slider("Batch Size", 8, 128, 32)
        
        # Optimizer selection
        st.sidebar.subheader("Select Optimizers")
        selected_optimizers = []
        for opt in optimizer_options:
            if st.sidebar.checkbox(opt, value=(opt in ['SGD', 'Adam']), key=f"moon_{opt}"):
                selected_optimizers.append(opt)
        
        if not selected_optimizers:
            st.warning("Please select at least one optimizer.")
            return
        
        # Prepare dataset
        st.header("Moon-shaped Dataset")
        X_train, X_test, y_train, y_test, X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, train_loader = prepare_moon_dataset(
            n_samples=n_samples, 
            noise=noise,
            batch_size=batch_size
        )
        
        # Plot dataset
        st.subheader("Dataset Visualization")
        st.pyplot(plot_moon_dataset(X_train, X_test, y_train, y_test))
        
        # Training button
        if st.button("Train Models"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"Using device: {device}")
            
            # Initialize results dict
            results = {}
            models = {}
            
            # Train each selected optimizer
            progress_text = "Training models..."
            my_bar = st.progress(0)
            
            for i, opt_name in enumerate(selected_optimizers):
                st.write(f"Training with {opt_name}...")
                model, loss_hist, acc_hist = train_moon_model(
                    optimizer_name=opt_name,
                    train_loader=train_loader,
                    X_test_tensor=X_test_tensor,
                    y_test_tensor=y_test_tensor,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                results[opt_name] = (loss_hist, acc_hist)
                models[opt_name] = model
                my_bar.progress((i + 1) / len(selected_optimizers))
            
            # Create columns for the results
            col1, col2 = st.columns(2)
            
            with col1:
                # Create animated progress visualization
                st.subheader("Training Progress Animation")
                progress_placeholder = st.empty()
                
                for epoch in range(epochs):
                    if epoch % 2 == 0 or epoch == epochs - 1:  # Animate every other epoch for speed
                        fig = create_animated_curves(results, epoch, "Moon Dataset")
                        progress_placeholder.pyplot(fig)
                        plt.close(fig)
                        time.sleep(0.1)
                
                # Final training curves
                st.subheader("Training Results")
                st.pyplot(plot_training_curves(results, "Moon Dataset"))
            
            with col2:
                # Decision Boundaries for each optimizer
                st.subheader("Decision Boundaries")
                
                for opt_name, model in models.items():
                    st.write(f"Decision Boundary for {opt_name}")
                    st.pyplot(plot_decision_boundary(model, X_test_tensor, y_test_tensor, device))
            
            # Final results table
            st.subheader("Final Results")
            
            final_results = {
                "Optimizer": [],
                "Final Loss": [],
                "Final Accuracy": [],
                "Convergence Time (epochs)": []
            }
            
            for opt_name, (loss_hist, acc_hist) in results.items():
                final_results["Optimizer"].append(opt_name)
                final_results["Final Loss"].append(f"{loss_hist[-1]:.6f}")
                final_results["Final Accuracy"].append(f"{acc_hist[-1]:.4f}")
                
                # Determine convergence time (when accuracy reaches 95% of final accuracy)
                thresh = 0.95 * acc_hist[-1]
                convergence_time = next((i for i, acc in enumerate(acc_hist) if acc >= thresh), epochs)
                final_results["Convergence Time (epochs)"].append(convergence_time)
            
            st.table(final_results)
            
    else:  # CIFAR-10 task
        # Show dataset parameters
        st.sidebar.subheader("Dataset Parameters")
        subset_percentage = st.sidebar.slider("Dataset Percentage", 0.05, 1.0, 0.2, step=0.05)
        batch_size = st.sidebar.slider("Batch Size", 32, 256, 128, step=32)
        
        # Show training parameters
        st.sidebar.subheader("Training Parameters")
        learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)
        epochs = st.sidebar.slider("Number of Epochs", 5, 30, 10)
        
        # Optimizer selection
        st.sidebar.subheader("Select Optimizers")
        selected_optimizers = []
        for opt in optimizer_options:
            if st.sidebar.checkbox(opt, value=(opt in ['SGD', 'Adam']), key=f"cifar_{opt}"):
                selected_optimizers.append(opt)
        
        if not selected_optimizers:
            st.warning("Please select at least one optimizer.")
            return
        
        # Prepare dataset
        st.header("CIFAR-10 Dataset")
        
        # Add warning about training time
        st.warning("""
        **Note:** Training ResNet18 on CIFAR-10 can take significant time, especially without a GPU.
        Consider using a smaller subset percentage and fewer epochs if running on CPU.
        """)
        
        # Try to load the dataset
        try:
            with st.spinner("Loading CIFAR-10 dataset..."):
                trainloader, testloader, classes = prepare_cifar10_dataset(
                    subset_percentage=subset_percentage,
                    batch_size=batch_size
                )
            
            # Show example images
            st.subheader("Example Images")
            st.pyplot(plot_cifar_examples(trainloader, classes))
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return
        
        # Training button
        if st.button("Train Models"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.write(f"Using device: {device}")
            
            # Initialize results dict
            results = {}
            models = {}
            
            # Create progress bar
            progress_text = "Training models..."
            my_bar = st.progress(0)
            
            # Train each selected optimizer
            for i, opt_name in enumerate(selected_optimizers):
                st.write(f"Training with {opt_name}...")
                model, loss_hist, acc_hist = train_cifar_model(
                    optimizer_name=opt_name,
                    trainloader=trainloader,
                    testloader=testloader,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                results[opt_name] = (loss_hist, acc_hist)
                models[opt_name] = model
                my_bar.progress((i + 1) / len(selected_optimizers))
            
            # Create columns for results visualization
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create animated progress visualization
                st.subheader("Training Progress Animation")
                progress_placeholder = st.empty()
                
                for epoch in range(epochs):
                    if epoch % 2 == 0 or epoch == epochs - 1:  # Animate every other epoch for speed
                        fig = create_animated_curves(results, epoch, "CIFAR-10")
                        progress_placeholder.pyplot(fig)
                        plt.close(fig)
                        time.sleep(0.1)
                
                # Final training curves
                st.subheader("Training Results")
                st.pyplot(plot_training_curves(results, "CIFAR-10"))
            
            with col2:
                # Final results table
                st.subheader("Final Results")
                
                final_results = {
                    "Optimizer": [],
                    "Final Loss": [],
                    "Final Accuracy": [],
                    "Best Accuracy": []
                }
                
                for opt_name, (loss_hist, acc_hist) in results.items():
                    final_results["Optimizer"].append(opt_name)
                    final_results["Final Loss"].append(f"{loss_hist[-1]:.6f}")
                    final_results["Final Accuracy"].append(f"{acc_hist[-1]:.4f}")
                    final_results["Best Accuracy"].append(f"{max(acc_hist):.4f}")
                
                st.table(final_results)
                
                # Optimizer recommendations based on results
                best_optimizer = max(results.items(), key=lambda x: x[1][1][-1])[0]
                st.success(f"**Best Performer**: {best_optimizer} achieved the highest final accuracy on CIFAR-10")
                
                # Performance summary
                st.subheader("Performance Summary")
                
                for opt_name in selected_optimizers:
                    loss_hist, acc_hist = results[opt_name]
                    
                    if opt_name == 'SGD':
                        st.markdown("- **SGD**: Slow but steady progress; benefits from longer training")
                    elif opt_name == 'Momentum':
                        st.markdown("- **Momentum**: Faster convergence than SGD; helps overcome plateaus")
                    elif opt_name == 'NAG':
                        st.markdown("- **Nesterov**: Similar to Momentum but with lookahead correction")
                    elif opt_name == 'Adam':
                        st.markdown("- **Adam**: Fast initial progress; adaptive learning rates per parameter")
                    else:
                        st.markdown(f"- **{opt_name}**: Final accuracy: {acc_hist[-1]:.4f}")

# Run the app
if __name__ == "__main__":
    main()