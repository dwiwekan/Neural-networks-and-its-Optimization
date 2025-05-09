import os
os.environ['STREAMLIT_SERVER_WATCH_EXCLUSIONS'] = 'torch'

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# Set Streamlit's page configuration (including layout and wide mode)
st.set_page_config(layout="wide")

# 1. First, fix the dataset generation function to create more separated classes
def generate_classification_data(n_samples=200, n_features=2, n_classes=2, random_state=42):
    n_informative = min(n_features, 2)  # Ensure n_informative doesn't exceed n_features
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=0,
        class_sep=2.0,           # Increase class separation
        n_clusters_per_class=1,
        random_state=random_state,
        flip_y=0.1             # Reduce noise
    )
    return X, y

# Neural Network class with configurable layers and activation functions
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation_function, regularization, reg_rate):
        super(SimpleNN, self).__init__()

        self.layers = []
        prev_dim = input_dim
        for i, neurons in enumerate(hidden_layers):
            self.layers.append(nn.Linear(prev_dim, neurons))
            if activation_function == 'ReLU':
                self.layers.append(nn.ReLU())
            elif activation_function == 'Sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation_function == 'Tanh':
                self.layers.append(nn.Tanh())
            prev_dim = neurons

        # Output layer
        self.layers.append(nn.Linear(prev_dim, 1))
        self.layers.append(nn.Sigmoid())  # Classification output

        # Apply regularization
        if regularization == 'L1':
            self.regularization = nn.L1Loss()
        else:  # L2 regularization
            self.regularization = nn.MSELoss()

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(X_train, y_train, X_val, y_val, epochs, learning_rate, model, criterion, optimizer):
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Convert data to tensors
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Regularization - FIXED
        reg_loss = 0.0
        if model.regularization == nn.L1Loss():
            # L1 regularization
            for param in model.parameters():
                reg_loss += torch.sum(torch.abs(param))
        else:
            # L2 regularization
            for param in model.parameters():
                reg_loss += torch.sum(param.pow(2))
        
        reg_loss *= 0.01  # Regularization strength
        total_loss = loss + reg_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # Validation loss and accuracy
        model.eval()
        val_inputs = torch.tensor(X_val, dtype=torch.float32)
        val_targets = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_targets)
        val_losses.append(val_loss.item())

        accuracy = accuracy_score(y_val, (val_outputs.detach().numpy() > 0.5).astype(int))
        accuracies.append(accuracy)

        # Logging
        train_losses.append(total_loss.item())

    return train_losses, val_losses, accuracies

# Add this function near your other visualization functions
def plot_neural_network(input_dim, hidden_layers, output_dim=1):
    """
    Creates a visual representation of the neural network architecture
    """
    # Remove any layers with zero neurons
    hidden_layers = [neurons for neurons in hidden_layers if neurons > 0]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Calculate the maximum number of neurons in any layer for scaling
    max_neurons = max([input_dim] + hidden_layers + [output_dim])
    
    # Calculate layer positions
    n_layers = len(hidden_layers) + 2  # input + hidden + output
    layer_positions = np.linspace(0, 1, n_layers)
    
    # Colors
    input_color = "#66c2a5"   # teal for input
    hidden_color = "#8da0cb"  # blue for hidden
    output_color = "#fc8d62"  # orange for output
    
    # Draw input layer
    draw_layer(ax, layer_positions[0], input_dim, max_neurons, "Input Layer", input_color)
    
    # Draw hidden layers
    for i, neurons in enumerate(hidden_layers):
        draw_layer(ax, layer_positions[i+1], neurons, max_neurons, f"Hidden {i+1}", hidden_color)
    
    # Draw output layer
    draw_layer(ax, layer_positions[-1], output_dim, max_neurons, "Output", output_color)
    
    # Draw connections between layers
    draw_connections(ax, layer_positions, [input_dim] + hidden_layers + [output_dim], max_neurons)
    
    # Remove axes
    ax.axis('off')
    ax.set_title("Neural Network Architecture")
    
    return fig

def draw_layer(ax, x_pos, n_neurons, max_neurons, label, color):
    """Helper function to draw a layer of neurons"""
    neuron_size = 0.02
    spacing = min(0.8 / max(max_neurons, 1), 0.1)  # Adjust spacing based on max neurons
    
    # Calculate total height of layer
    layer_height = (n_neurons - 1) * spacing
    
    # Draw each neuron
    for i in range(n_neurons):
        # Calculate y position for the neuron
        if n_neurons == 1:
            y_pos = 0.5
        else:
            y_pos = 0.5 + layer_height/2 - i * spacing
        
        # Draw the neuron as a circle
        circle = plt.Circle((x_pos, y_pos), neuron_size, color=color, ec='black', zorder=4)
        ax.add_patch(circle)
    
    # Add layer label
    ax.text(x_pos, 0.1, label, ha='center', va='center', fontsize=10, fontweight='bold')

def draw_connections(ax, layer_positions, neurons_per_layer, max_neurons):
    """Helper function to draw connections between layers"""
    for i in range(len(neurons_per_layer) - 1):
        spacing_left = min(0.8 / max(max_neurons, 1), 0.1)
        spacing_right = min(0.8 / max(max_neurons, 1), 0.1)
        
        for j in range(neurons_per_layer[i]):
            # Left neuron position
            if neurons_per_layer[i] == 1:
                y_left = 0.5
            else:
                layer_height_left = (neurons_per_layer[i] - 1) * spacing_left
                y_left = 0.5 + layer_height_left/2 - j * spacing_left
            
            for k in range(neurons_per_layer[i+1]):
                # Right neuron position
                if neurons_per_layer[i+1] == 1:
                    y_right = 0.5
                else:
                    layer_height_right = (neurons_per_layer[i+1] - 1) * spacing_right
                    y_right = 0.5 + layer_height_right/2 - k * spacing_right
                
                # Draw the connection line
                ax.plot([layer_positions[i], layer_positions[i+1]], 
                        [y_left, y_right], 'k-', alpha=0.1, linewidth=0.5)
                
def plot_decision_boundary(X, y, model, ax=None):
    # Ensure we're only using the first two features for visualization
    X_vis = X[:, :2] if X.shape[1] > 2 else X
    
    # Create a mesh grid on which we will run our model
    h = 0.01  # Step size of the mesh
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Create a tensor of appropriate shape
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # If the model expects more features, pad with zeros
    if X.shape[1] > 2:
        padding = np.zeros((mesh_points.shape[0], X.shape[1] - 2))
        mesh_points = np.hstack((mesh_points, padding))
    
    # Convert to PyTorch tensors
    mesh_inputs = torch.tensor(mesh_points, dtype=torch.float32)
    
    # Predict class labels for each point in the mesh
    model.eval()
    with torch.no_grad():
        mesh_outputs = model(mesh_inputs).detach().numpy()
    
    # Reshape the predictions back to the mesh grid shape
    mesh_predictions = mesh_outputs.reshape(xx.shape)
    
    # Use provided axis or create a new figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    
    # Plot the decision boundary
    ax.contourf(xx, yy, mesh_predictions > 0.5, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot the original data points
    scatter = ax.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    
    # Add labels and title
    ax.set_title('Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return fig

# Main Function for Streamlit ANN Basics Page
def main():
    # Inside your main() function, at the beginning:
    if 'X' not in st.session_state or 'y' not in st.session_state:
        # Initialize with default dataset if not already in session state
        X, y = generate_classification_data()
        st.session_state['X'] = X
        st.session_state['y'] = y
    st.title("ANN Basics: Interactive Classification Model")

    # Add an introductory paragraph
    st.markdown("""
    Artificial Neural Networks (ANNs) are computing systems inspired by the biological neural networks in human brains. 
    This interactive model demonstrates how ANNs learn decision boundaries to classify data points. Experiment with 
    different network architectures, activation functions, and hyperparameters to see how they affect the learning 
    process and classification accuracy.
    """)

    # Creating three columns for the layout
    col1, col2, col3 = st.columns([2, 3, 2])

    # Column 1: Generate Dataset and Show Code
    # In Column 1 section
    with col1:
        st.subheader("Generate Dataset")
        
        # Sliders for dataset parameters
        n_samples = st.slider("Select number of samples", 100, 1000, 200)
        random_state = st.slider("Select random state", 0, 100, 42)

        # Button to generate dataset
        if st.button("Generate Dataset"):
            X, y = generate_classification_data(n_samples, random_state=random_state)
            # Store in session state for later use
            st.session_state['X'] = X
            st.session_state['y'] = y
            
            # Store visualization parameters in session state
            st.session_state['dataset_generated'] = True
            st.success("Dataset Generated!")
        
        # Always show visualization if dataset exists
        if 'X' in st.session_state and 'y' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=50, edgecolors='k')
            ax.set_title('2D Classification Dataset')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            plt.close(fig)

    # Column 2: Neural Network Setup
    with col2:
        st.subheader("Neural Network Setup")
        
        st.markdown("""
        The neural network architecture and training parameters determine how well the model learns to classify the data.
        Experiment with different settings to see how they impact performance.
        """)
        
        # Hyperparameters for NN
        st.markdown("**Training Parameters:**")
        epochs = st.slider("Select number of epochs (training iterations)", 2, 2000, 100)
        st.caption("More epochs usually improve learning but may lead to overfitting")
        
        learning_rate = st.selectbox("Select learning rate (step size during training)", 
                                [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 1, 3, 10])
        st.caption("Controls how quickly the model adapts to the problem; too high may cause instability")
        
        activation_function = st.selectbox("Select activation function (adds non-linearity)", 
                                        ['ReLU', 'Sigmoid', 'Tanh', 'Linear'])
        st.caption("ReLU is often fastest, Sigmoid/Tanh work well for bounded outputs")
        
        regularization = st.selectbox("Select regularization (prevents overfitting)", 
                                    ['None', 'L1', 'L2'])
        st.caption("L1 produces sparse weights, L2 keeps weights small")
        
        reg_rate = st.selectbox("Select regularization rate (strength)", 
                            [0.001, 0.003, 0.01, 0.03, 0.1, 1, 3])
        st.caption("Higher values increase regularization effect")
        
        # Network architecture
        st.markdown("**Network Architecture:**")
        st.caption("Configure the size of each hidden layer (0 neurons removes the layer)")
        hidden_layers = []
        for i in range(4):
            neurons = st.slider(f"Hidden Layer {i+1} size", min_value=2, max_value=128, value=32, step=2)
            hidden_layers.append(neurons)

        # In Column 2 section, after hidden layers setup
        st.write(f"Selected hidden layers configuration: {hidden_layers}")

        # Filter out layers with 0 neurons for the visualization
        active_layers = [n for n in hidden_layers if n > 0]

        # Neural network visualization
        st.subheader("Neural Network Visualization")
        nn_fig = plot_neural_network(input_dim=2, hidden_layers=active_layers)
        st.pyplot(nn_fig)
        plt.close(nn_fig)  # Close to prevent memory leaks
    # Column 3: Dataset Visualization and Training
    # Column 3: Dataset Visualization and Training
    # In your Column 3 section:
    with col3:
        # Get data from session state
        X = st.session_state['X']
        y = st.session_state['y']
        
        # Train/Test split and Standardize Data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Scale the entire dataset for visualization consistency
        X_scaled = scaler.transform(X)
        
        # Model Setup
        hidden_layers = [neurons for neurons in hidden_layers if neurons > 0]
        model = SimpleNN(input_dim=X_train_scaled.shape[1], hidden_layers=hidden_layers,
                        activation_function=activation_function, regularization=regularization, reg_rate=reg_rate)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create placeholders
        plot_placeholder = st.empty()
        progress_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Begin training with button
        start_button = st.button("Begin Training", help="Click to start training the model with the selected parameters.", key="start_button", disabled='dataset_generated' not in st.session_state or not st.session_state['dataset_generated'])
        
        # In Column 3, modify the training section
        if start_button:
            # Initialize lists to store metrics across all epochs
            all_train_losses = []
            all_val_losses = []
            all_accuracies = []
            
            try:
                for epoch in range(epochs):
                    # Training loop
                    train_losses, val_losses, accuracies = train_model(
                        X_train_scaled, y_train, X_val_scaled, y_val, 1, learning_rate, model, criterion, optimizer
                    )
                    
                    # Collect metrics
                    all_train_losses.append(train_losses[-1])
                    all_val_losses.append(val_losses[-1]) 
                    all_accuracies.append(accuracies[-1])
                    
                    # Create figure for decision boundary plot
                    fig, ax = plt.subplots()
                    plot_decision_boundary(X_scaled, y, model, ax)
                    plot_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    # Update progress and metrics
                    progress_placeholder.progress((epoch + 1) / epochs)
                    metrics_placeholder.text(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Loss: {train_losses[-1]:.4f} | "
                        f"Val Loss: {val_losses[-1]:.4f} | "
                        f"Accuracy: {accuracies[-1]:.2f}"
                    )
                    
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
            
            # Create training/validation metrics plot
            st.subheader("Training Metrics")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
            
            # Plot losses
            epochs_range = range(1, len(all_train_losses) + 1)
            ax1.plot(epochs_range, all_train_losses, 'b-', label='Training Loss')
            ax1.plot(epochs_range, all_val_losses, 'r-', label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(epochs_range, all_accuracies, 'g-', label='Validation Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
            plt.close(fig)


if __name__ == "__main__":
    main()
