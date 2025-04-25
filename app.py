import streamlit as st
from nn.model import CNN
from nn.train import train_model
from utils.data_processing import load_fashion_mnist
from utils.plotting import plot_loss, plot_accuracy
import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    st.title("👗 Fashion MNIST Classification with CNN")
    st.markdown("""
        This application demonstrates a **Convolutional Neural Network (CNN)** trained to classify images from the **Fashion MNIST dataset**.
        
        You can adjust the hyperparameters from the sidebar and train the model directly from this interface.
    """)

    # Sidebar options for customization
    st.sidebar.header("Model Settings")
    epochs = st.sidebar.slider("Epochs", 1, 20, 5)
    learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=1)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)

    if st.sidebar.button("Start Training 🚀"):
        # Load Fashion MNIST dataset
        train_loader, test_loader, class_names = load_fashion_mnist(batch_size)

        # Instantiate CNN model
        model = CNN()

        # Create placeholders for live updating
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_chart_placeholder = st.empty()
        accuracy_chart_placeholder = st.empty()

        # Train the model and capture history
        history = {"train_loss": [], "test_accuracy": []}
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate_model(model, test_loader)

            history["train_loss"].append(avg_loss)
            history["test_accuracy"].append(accuracy)

            # Update live progress and charts
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

            # Update loss and accuracy charts separately
            loss_chart_placeholder.pyplot(plot_loss(history["train_loss"]))
            accuracy_chart_placeholder.pyplot(plot_accuracy(history["test_accuracy"]))

        st.success("🎉 Training Completed Successfully!")

        # Display predictions from test set
        st.subheader("Sample Predictions")
        plot_sample_predictions(model, test_loader, class_names)

# Evaluate function to check accuracy
def evaluate_model(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (correct / total) * 100

# Plotting sample predictions
def plot_sample_predictions(model, test_loader, class_names, num_images=6):
    images, labels = next(iter(test_loader))
    model.eval()
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    for idx in range(num_images):
        ax = axes[idx]
        img = images[idx].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
