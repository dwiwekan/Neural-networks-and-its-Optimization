import os
os.environ['STREAMLIT_SERVER_WATCH_EXCLUSIONS'] = 'torch'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="Gradient Descent Variants")

# Title and introduction
st.title("Gradient Descent Variants: Interactive Visualization")
st.markdown("""
Gradient descent is an optimization algorithm used to minimize a function by iteratively moving 
in the direction of steepest descent. This app demonstrates how different gradient descent variants behave
on a simple convex function:

- **Full Batch Gradient Descent**: Uses exact gradients (smooth path)
- **Mini-batch Gradient Descent**: Uses subset of points for gradient calculation (slight noise)
- **Stochastic Gradient Descent**: Uses single random point (noisy path)

Observe how these methods navigate the same cost surface differently!
""")

# Create and define our cost function (simplified to only use parabola)
def get_cost_function():
    def cost_func(x, y):
        return x**2 + y**2
    
    def gradient(x, y):
        return np.array([2*x, 2*y])
    
    x_range = (-5, 5)
    y_range = (-5, 5)
    global_min = (0, 0)
    
    return cost_func, gradient, x_range, y_range, global_min

# Create a surface plot of the cost function
def plot_cost_function(cost_func, x_range, y_range, paths=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points to evaluate the function
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(cost_func)(X, Y)
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)
    
    # Add a colorbar
    if hasattr(ax, 'figure'):
        fig = ax.figure
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Cost')
    ax.set_title('Cost Function Surface')
    
    # Plot optimization path if provided
    if paths is not None:
        colors = ['r', 'g', 'b']  # Colors for different paths
        for idx, (path, label) in enumerate(paths):
            if len(path) > 0:
                path_x, path_y = zip(*path)
                path_z = [cost_func(x, y) for x, y in zip(path_x, path_y)]
                color = colors[idx % len(colors)]
                ax.plot(path_x, path_y, path_z, color=color, linewidth=2.5, label=label)
                ax.plot([path_x[-1]], [path_y[-1]], [path_z[-1]], 'o', color=color, markersize=7)
    
    if paths:
        ax.legend()
    
    return ax

# Create a contour plot of the cost function
def plot_contour(cost_func, x_range, y_range, paths=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(cost_func)(X, Y)
    
    # Create contour plot
    contour = ax.contourf(X, Y, Z, levels=50, cmap='coolwarm', alpha=0.8)
    ax.contour(X, Y, Z, levels=20, colors='k', alpha=0.3, linewidths=0.5)
    
    # Add a colorbar
    plt.colorbar(contour, ax=ax)
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Cost Function Contours with Gradient Descent Path')
    
    # Plot optimization path if provided
    if paths is not None:
        colors = ['r', 'g', 'b']  # Colors for different paths
        for idx, (path, label) in enumerate(paths):
            if len(path) > 0:
                path_x, path_y = zip(*path)
                color = colors[idx % len(colors)]
                ax.plot(path_x, path_y, color=color, linewidth=2.5, label=label)
                ax.plot([path_x[-1]], [path_y[-1]], 'o', color=color, markersize=7)
        
        # Add legend
        ax.legend()
    
    return ax

# Implementation of different gradient descent variants
def full_batch_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum=0):
    """Full batch gradient descent - uses exact gradients"""
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    
    # For momentum calculation
    prev_step_x, prev_step_y = 0, 0
    
    for _ in range(iterations):
        # Calculate exact gradient
        grad = gradient_func(x, y)
        
        # Apply momentum
        step_x = learning_rate * grad[0] + momentum * prev_step_x
        step_y = learning_rate * grad[1] + momentum * prev_step_y
        
        # Update position
        x = x - step_x
        y = y - step_y
        
        # Store for momentum
        prev_step_x = step_x
        prev_step_y = step_y
        
        # Add to path
        path.append((x, y))
    
    return path

def mini_batch_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum=0, batch_size=5):
    """Mini-batch gradient descent - simulates using batches of points"""
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    
    # For momentum calculation
    prev_step_x, prev_step_y = 0, 0
    
    # Use the provided batch_size parameter
    
    for _ in range(iterations):
        # Simulate mini-batch gradient with some noise
        true_grad = gradient_func(x, y)
        
        # Add some random noise to simulate mini-batch behavior
        noise_level = 0.2 * np.linalg.norm(true_grad) / np.sqrt(batch_size)
        noise_x = np.random.normal(0, noise_level)
        noise_y = np.random.normal(0, noise_level)
        
        grad = np.array([true_grad[0] + noise_x, true_grad[1] + noise_y])
        
        # Apply momentum
        step_x = learning_rate * grad[0] + momentum * prev_step_x
        step_y = learning_rate * grad[1] + momentum * prev_step_y
        
        # Update position
        x = x - step_x
        y = y - step_y
        
        # Store for momentum
        prev_step_x = step_x
        prev_step_y = step_y
        
        # Add to path
        path.append((x, y))
    
    return path

def stochastic_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum=0):
    """Stochastic gradient descent - simulates using single points with high variance"""
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    
    # For momentum calculation
    prev_step_x, prev_step_y = 0, 0
    
    for _ in range(iterations):
        # Get the true gradient
        true_grad = gradient_func(x, y)
        
        # Add significant noise to simulate SGD behavior (single sample)
        noise_level = 0.5 * np.linalg.norm(true_grad)
        noise_x = np.random.normal(0, noise_level)
        noise_y = np.random.normal(0, noise_level)
        
        grad = np.array([true_grad[0] + noise_x, true_grad[1] + noise_y])
        
        # Apply momentum
        step_x = learning_rate * grad[0] + momentum * prev_step_x
        step_y = learning_rate * grad[1] + momentum * prev_step_y
        
        # Update position
        x = x - step_x
        y = y - step_y
        
        # Store for momentum
        prev_step_x = step_x
        prev_step_y = step_y
        
        # Add to path
        path.append((x, y))
    
    return path

# Main function
def main():
    # Create sidebar for parameters
    st.sidebar.header("Gradient Descent Parameters")
    
    # Get the cost function (always parabola)
    cost_func, gradient_func, x_range, y_range, global_min = get_cost_function()
    
    # Starting position
    st.sidebar.subheader("Starting Position")
    start_x = st.sidebar.slider("X Starting Point", 
                            float(x_range[0]), 
                            float(x_range[1]), 
                            float(4.0),  # Start from (4,4) to better visualize the paths
                            step=0.1)
    
    start_y = st.sidebar.slider("Y Starting Point", 
                            float(y_range[0]), 
                            float(y_range[1]), 
                            float(4.0),
                            step=0.1)
    
    # Optimization parameters
    st.sidebar.subheader("Optimization Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01, step=0.001)  # Changed max to 1.0 and default to 0.01
    momentum = st.sidebar.slider("Momentum", 0.0, 0.99, 0.0, step=0.01)
    iterations = st.sidebar.slider("Maximum Iterations", 5, 200, 50)
    
    # Add mini-batch size slider
    batch_size = st.sidebar.slider("Mini-batch Size", 2, 50, 5, step=1, 
                               help="Number of samples used in each mini-batch update")
    
    # Show global minimum
    st.sidebar.subheader("Function Information")
    st.sidebar.write(f"Global Minimum: {global_min}")
    
    # Create a two-column layout for the plots
    col1, col2 = st.columns(2)
    
    # Interactive visualizations for static view
    with col1:
        st.subheader("3D Surface View")
        st.markdown("""
        This 3D surface shows the cost function f(x,y) = x² + y². The global minimum is at (0,0).
        
        - **Red points**: Full Batch GD path
        - **Green points**: Mini-batch GD path
        - **Blue points**: Stochastic GD path
        """)
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        plot_cost_function(cost_func, x_range, y_range, ax=ax1)
        
        # Mark the global minimum
        ax1.scatter([global_min[0]], [global_min[1]], [cost_func(*global_min)], color='yellow', s=100, label='Global Minimum')
        ax1.legend()
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Contour View")
        st.markdown("""
        This contour plot shows level curves of the cost function. Each curve connects points with the same cost value.
        
        The optimization algorithms will try to find a path from the starting point (red dot) to the global minimum (center).
        """)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_contour(cost_func, x_range, y_range, ax=ax2)
        
        # Mark the starting point and global minimum
        ax2.plot([start_x], [start_y], 'ro', markersize=10, label='Starting point')
        ax2.plot([global_min[0]], [global_min[1]], 'yo', markersize=10, label='Global minimum')
        ax2.legend()
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Add a checkbox to select which methods to compare
    st.subheader("Select Gradient Descent Variants to Compare")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        full_batch = st.checkbox("Full Batch GD", value=True)
    with col2:
        mini_batch = st.checkbox("Mini-batch GD", value=True)
    with col3:
        stochastic = st.checkbox("Stochastic GD", value=True)
    
    # Button to start animation
    if st.button("Start Gradient Descent"):
        with st.spinner("Running gradient descent..."):
            # Track selected paths
            paths = []
            costs_history = []
            labels = []
            
            # Run selected gradient descent variants
            if full_batch:
                path_full = full_batch_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum)
                paths.append((path_full, "Full Batch GD"))
                costs_full = [cost_func(x, y) for x, y in path_full]
                costs_history.append(costs_full)
                labels.append("Full Batch GD")
            
            if mini_batch:
                path_mini = mini_batch_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum, batch_size)
                paths.append((path_mini, "Mini-batch GD"))
                costs_mini = [cost_func(x, y) for x, y in path_mini]
                costs_history.append(costs_mini)
                labels.append(f"Mini-batch GD (batch={batch_size})")
            
            if stochastic:
                path_sgd = stochastic_gd(start_x, start_y, gradient_func, learning_rate, iterations, momentum)
                paths.append((path_sgd, "Stochastic GD"))
                costs_sgd = [cost_func(x, y) for x, y in path_sgd]
                costs_history.append(costs_sgd)
                labels.append("Stochastic GD")
            
            # Create animation placeholders
            progress_placeholder = st.progress(0)
            anim_placeholder = st.empty()
            
            # Prepare for animation
            anim_steps = min(iterations, 50)  # Limit animation frames for performance
            
            # Animate the gradient descent process
            for i in range(anim_steps):
                # Update progress bar
                progress_placeholder.progress((i + 1) / anim_steps)
                
                # Calculate subset of paths for this frame
                current_paths = []
                for idx, (path, label) in enumerate(paths):
                    frame_index = min(int((i + 1) * len(path) / anim_steps), len(path) - 1)
                    current_paths.append((path[:frame_index+1], label))
                
                # Create figure with two subplots - contour and costs
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot contour with current paths
                plot_contour(cost_func, x_range, y_range, paths=current_paths, ax=ax1)
                
                # Plot costs vs iterations
                for idx, costs in enumerate(costs_history):
                    frame_index = min(int((i + 1) * len(costs) / anim_steps), len(costs) - 1)
                    iterations_list = list(range(frame_index + 1))
                    ax2.plot(iterations_list, costs[:frame_index+1], 
                             label=labels[idx], 
                             color=['red', 'green', 'blue'][idx % 3])
                    
                ax2.set_xlabel('Iterations')
                ax2.set_ylabel('Cost')
                ax2.set_title('Cost vs Iterations')
                ax2.grid(True)
                ax2.legend()
                
                # Display the figure
                anim_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Add small delay
                time.sleep(0.1)
            

            
            # Display color-coded legend for methods
            st.markdown("""
            ### Method Characteristics:
            
            <style>
            .legend-item {
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }
            .color-box {
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border: 1px solid black;
            }
            </style>
            
            <div class="legend-item">
                <div class="color-box" style="background-color: red;"></div>
                <strong>Full Batch GD</strong>: Stable, direct path. Uses exact gradients.
            </div>
            <div class="legend-item">
                <div class="color-box" style="background-color: green;"></div>
                <strong>Mini-batch GD</strong>: Slightly noisy path. Uses batches of samples.
            </div>
            <div class="legend-item">
                <div class="color-box" style="background-color: blue;"></div>
                <strong>Stochastic GD</strong>: Very noisy path. Uses single random samples.
            </div>
            """, unsafe_allow_html=True)
            
            # Display final stats
            if paths:
                st.subheader("Final Results")
                results = {"Method": [], "Final X": [], "Final Y": [], "Final Cost": [], "Distance from Optimum": []}
                
                for (path, label) in paths:
                    if path:
                        final_x, final_y = path[-1]
                        final_cost = cost_func(final_x, final_y)
                        distance = np.sqrt((final_x - global_min[0])**2 + (final_y - global_min[1])**2)
                        
                        results["Method"].append(label)
                        results["Final X"].append(f"{final_x:.6f}")
                        results["Final Y"].append(f"{final_y:.6f}")
                        results["Final Cost"].append(f"{final_cost:.6f}")
                        results["Distance from Optimum"].append(f"{distance:.6f}")
                
                st.table(results)

# Run the app
if __name__ == "__main__":
    main()