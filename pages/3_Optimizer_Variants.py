import os
os.environ['STREAMLIT_SERVER_WATCH_EXCLUSIONS'] = 'torch'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="Optimizer Variants")

# Title and introduction
st.title("Optimizer Variants: Interactive Visualization")
st.markdown("""
This app demonstrates how different optimization algorithms navigate loss landscapes. You can compare
how each algorithm behaves on different types of functions:

- **Convex Function**: A bowl-shaped function with a single global minimum
- **Saddle Point Function**: A function with a saddle point (min in one direction, max in another)

Observe how advanced optimizers like Adam, RMSprop, and others compare to basic SGD!
""")

# Define loss functions and their gradients
def saddle_loss_fn(x, y):
    return x**2 - y**2

def grad_saddle(x):
    return np.array([2 * x[0], -2 * x[1]])

def convex_loss_fn(x, y):
    return x**2 + 10*y**2

def grad_convex(x):
    return np.array([2 * x[0], 20 * x[1]])

# Optimizer implementations
def sgd(x_start, step, grad_fn, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    for _ in range(iteration):
        grad = grad_fn(x)
        x -= step * grad
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def momentum(x_start, step, grad_fn, discount=0.9, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    v = np.zeros_like(x)
    for _ in range(iteration):
        grad = grad_fn(x)
        v = discount * v + grad
        x -= step * v
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def nesterov(x_start, step, grad_fn, discount=0.9, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    v = np.zeros_like(x)
    for _ in range(iteration):
        future_x = x - step * discount * v
        grad = grad_fn(future_x)
        v = discount * v + grad
        x -= step * v
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def adagrad(x_start, step, grad_fn, delta=1e-8, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    G = np.zeros_like(x)
    for _ in range(iteration):
        grad = grad_fn(x)
        G += grad**2
        x -= step * grad / (np.sqrt(G) + delta)
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def adadelta(x_start, step, grad_fn, rho=0.95, delta=1e-6, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    Eg = np.zeros_like(x)
    Edx = np.zeros_like(x)
    for _ in range(iteration):
        grad = grad_fn(x)
        Eg = rho * Eg + (1 - rho) * grad**2
        dx = - (np.sqrt(Edx + delta) / np.sqrt(Eg + delta)) * grad
        x += dx
        Edx = rho * Edx + (1 - rho) * dx**2
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def rmsprop(x_start, step, grad_fn, rho=0.9, delta=1e-8, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    Eg = np.zeros_like(x)
    for _ in range(iteration):
        grad = grad_fn(x)
        Eg = rho * Eg + (1 - rho) * grad**2
        x -= step * grad / (np.sqrt(Eg) + delta)
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

def adam(x_start, step, grad_fn, beta1=0.9, beta2=0.999, delta=1e-8, iteration=50):
    x = np.array(x_start, dtype='float64')
    trace = [x.copy()]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iteration + 1):
        grad = grad_fn(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= step * m_hat / (np.sqrt(v_hat) + delta)
        trace.append(x.copy())
        if np.linalg.norm(grad) < 1e-6:
            break
    return np.array(trace)

# Dictionary of optimizer functions and colors
optimizer_functions = {
    "SGD": sgd,
    "Momentum": momentum,
    "Nesterov": nesterov,
    "AdaGrad": adagrad,
    "AdaDelta": adadelta,
    "RMSProp": rmsprop,
    "Adam": adam
}

optimizer_colors = {
    "SGD": "#1f77b4",
    "Momentum": "#ff7f0e",
    "Nesterov": "#2ca02c",
    "AdaGrad": "#d62728",
    "AdaDelta": "#9467bd",
    "RMSProp": "#8c564b",
    "Adam": "#e377c2"
}

# Visualization functions
def plot_contour(ax, loss_fn, x_range=(-8, 8), y_range=(-8, 8)):
    """Draw contour lines for the given loss function"""
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)
    Z = loss_fn(X, Y)
    ax.contour(X, Y, Z, levels=20, colors='gray', linewidths=0.7)
    return ax

def plot_trajectory(ax, trace, name, loss_fn=None):
    """Plot optimization trajectory with arrows"""
    color = optimizer_colors[name]
    ax.plot(trace[:, 0], trace[:, 1], color=color, linewidth=2, alpha=0.9)
    ax.scatter(trace[:, 0], trace[:, 1], color=color, s=20, alpha=0.6)
    
    # Add arrows to show direction of movement
    stride = max(1, len(trace)//10)  # Show every 10th step or less
    for i in range(0, len(trace)-1, stride):
        start = trace[i]
        end = trace[i+1]
        dx, dy = end - start
        ax.arrow(start[0], start[1], dx, dy,
                 head_width=0.3, head_length=0.4, fc=color, ec=color,
                 length_includes_head=True, alpha=0.9)
    
    # Calculate loss values for trajectory if loss function is provided
    if loss_fn is not None:
        losses = [loss_fn(p[0], p[1]) for p in trace]
        return losses
    return None

def plot_loss_curves(ax, losses_dict):
    """Plot loss vs iteration for multiple optimizers"""
    for name, losses in losses_dict.items():
        ax.plot(losses, label=name, color=optimizer_colors[name], linewidth=2)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss vs Iterations')
    ax.grid(True)
    ax.legend()

# Main function
def main():
    # Sidebar for controls
    st.sidebar.header("Optimizer Settings")
    
    # Choose loss function
    loss_type = st.sidebar.selectbox(
        "Select Loss Function Type",
        ["Convex Function", "Saddle Point Function"]
    )
    
    # Set the appropriate loss and gradient function based on selection
    if loss_type == "Convex Function":
        loss_fn = convex_loss_fn
        grad_fn = grad_convex
        start_x = 7.0
        start_y = 5.0
        surface_title = "Convex Function (f(x,y) = x² + 10y²)"
        description = """
        **Convex Function**: This is a bowl-shaped function with a single global minimum at (0,0).
        
        All optimizers should eventually converge to this minimum, but with different paths and speeds.
        """
    else:
        loss_fn = saddle_loss_fn
        grad_fn = grad_saddle
        start_x = 7.0
        start_y = 0.001
        surface_title = "Saddle Point Function (f(x,y) = x² - y²)"
        description = """
        **Saddle Point Function**: This function has a saddle point at (0,0) - a minimum in the x direction and a maximum in the y direction.
        
        This type of function is challenging for optimizers because the gradient can be misleading near the saddle point.
        """
    
    # Optimization parameters
    st.sidebar.subheader("Starting Point")
    x_start = st.sidebar.slider("X Starting Point", -7.0, 7.0, start_x, step=0.1)
    y_start = st.sidebar.slider("Y Starting Point", -7.0, 7.0, start_y, step=0.1)
    
    st.sidebar.subheader("Algorithm Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, step=0.01)
    iterations = st.sidebar.slider("Maximum Iterations", 10, 200, 50)
    momentum_beta = st.sidebar.slider("Momentum/Nesterov Beta", 0.5, 0.99, 0.9, step=0.01)
    adam_beta1 = st.sidebar.slider("Adam Beta1", 0.5, 0.99, 0.9, step=0.01)
    adam_beta2 = st.sidebar.slider("Adam Beta2", 0.9, 0.999, 0.999, step=0.001)
    
    # Select optimizers to compare
    st.sidebar.subheader("Select Optimizers to Compare")
    selected_optimizers = {}
    for opt_name in optimizer_functions.keys():
        if st.sidebar.checkbox(opt_name, value=True):
            selected_optimizers[opt_name] = optimizer_functions[opt_name]
    
    # Description and 3D visualization of loss function
    st.header(surface_title)
    st.markdown(description)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # 3D Surface visualization
    with col1:
        st.subheader("3D Surface View")
        
        fig1 = plt.figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        
        # Create grid
        x_range = (-8, 8)
        y_range = (-8, 8)
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = loss_fn(X, Y)
        
        # Plot surface
        surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8, linewidth=0, antialiased=True)
        
        # Mark starting point
        ax1.scatter([x_start], [y_start], [loss_fn(x_start, y_start)], color='red', s=100, label='Starting Point')
        
        # Mark global minimum for convex function
        if loss_type == "Convex Function":
            ax1.scatter([0], [0], [loss_fn(0, 0)], color='green', s=100, label='Global Minimum')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Loss Value')
        ax1.set_title('Loss Function Surface')
        ax1.legend()
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    # 2D Contour visualization
    with col2:
        st.subheader("Contour View")
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        plot_contour(ax2, loss_fn)
        
        # Mark starting point
        ax2.scatter([x_start], [y_start], color='red', s=100, label='Starting Point')
        
        # Mark global minimum for convex function
        if loss_type == "Convex Function":
            ax2.scatter([0], [0], color='green', s=100, label='Global Minimum')
            
        ax2.set_xlim(-8, 8)
        ax2.set_ylim(-8, 8)
        ax2.set_aspect('equal')
        ax2.set_title('Loss Function Contours')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.legend()
        
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Button to run optimizers
    if st.button("Run Optimizers"):
        if not selected_optimizers:
            st.warning("Please select at least one optimizer.")
        else:
            with st.spinner("Running optimization algorithms..."):
                # Container for trajectories and losses
                all_traces = {}
                all_losses = {}
                
                # Run each selected optimizer
                for name, opt_fn in selected_optimizers.items():
                    if name in ["Momentum", "Nesterov"]:
                        trace = opt_fn([x_start, y_start], learning_rate, grad_fn, 
                                    discount=momentum_beta, iteration=iterations)
                    elif name == "Adam":
                        trace = opt_fn([x_start, y_start], learning_rate, grad_fn, 
                                    beta1=adam_beta1, beta2=adam_beta2, iteration=iterations)
                    else:
                        trace = opt_fn([x_start, y_start], learning_rate, grad_fn, iteration=iterations)
                    
                    all_traces[name] = trace
                    all_losses[name] = [loss_fn(p[0], p[1]) for p in trace]
            
                # Create two columns for results display
                results_col1, results_col2 = st.columns([2, 2])

                with results_col1:
                    # Display both animations in the first column
                    st.subheader("Optimization Process")
                    progress_placeholder = st.progress(0)
                    
                    # Create placeholder for animation
                    anim_placeholder = st.empty()

                with results_col2:
                    # Just create a placeholder for results - we'll fill it after animation completes
                    st.subheader("Final Results")
                    results_table_placeholder = st.empty()
                    optimizer_desc_placeholder = st.empty()

                # Determine max steps for animation (limit to 50 frames)
                max_length = max([len(trace) for trace in all_traces.values()])
                anim_steps = min(50, max_length)

                # Animate the optimization process
                for i in range(anim_steps):
                    # Update progress bar
                    progress_placeholder.progress((i + 1) / anim_steps)
                    
                    # Create a figure with two subplots stacked vertically
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                    
                    # Top subplot: Optimization path
                    plot_contour(ax1, loss_fn)
                    ax1.scatter([x_start], [y_start], color='red', s=100, label='Starting Point')
                    
                    # For each optimizer, plot trajectory up to current step
                    for name, trace in all_traces.items():
                        # Calculate which index to use (scale to animation length)
                        idx = min(int((i + 1) * len(trace) / anim_steps), len(trace) - 1)
                        current_trace = trace[:idx+1]
                        ax1.plot(current_trace[:, 0], current_trace[:, 1], 
                                color=optimizer_colors[name], linewidth=2, alpha=0.9)
                        ax1.scatter(current_trace[-1, 0], current_trace[-1, 1], 
                                color=optimizer_colors[name], s=80, label=f"{name}")
                    
                    ax1.set_xlim(-8, 8)
                    ax1.set_ylim(-8, 8)
                    ax1.set_aspect('equal')
                    ax1.set_title(f'Optimization Path (Step {i+1}/{anim_steps})', fontsize=14)
                    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
                    
                    # Bottom subplot: Loss vs iterations
                    for name, losses in all_losses.items():
                        # Calculate which index to use
                        idx = min(int((i + 1) * len(losses) / anim_steps), len(losses) - 1)
                        current_losses = losses[:idx+1]
                        iterations_range = range(len(current_losses))
                        ax2.plot(iterations_range, current_losses, 
                                color=optimizer_colors[name], linewidth=2, label=name)
                        ax2.scatter(len(current_losses)-1, current_losses[-1], 
                                color=optimizer_colors[name], s=80)
                    
                    ax2.set_xlabel('Iterations')
                    ax2.set_ylabel('Loss Value')
                    ax2.set_title(f'Loss vs Iterations (Step {i+1}/{anim_steps})')
                    ax2.grid(True)
                    
                    # Use log scale if values have high variation
                    if any(max(losses) / (min(losses) + 1e-10) > 100 for losses in all_losses.values()):
                        ax2.set_yscale('log')
                    
                    ax2.legend()
                    plt.tight_layout()
                    
                    # Update the animation in column 1
                    anim_placeholder.pyplot(fig)
                    plt.close(fig)
                    
                    # Add small delay
                    time.sleep(0.1)

                # NOW fill column 2 with results AFTER animation is complete
                with results_col2:
                    # Prepare the results table
                    results_data = {
                        "Optimizer": [],
                        "Final X": [],
                        "Final Y": [],
                        "Final Loss": [],
                        "Steps": [],
                    }
                    
                    for name, trace in all_traces.items():
                        final_x, final_y = trace[-1]
                        results_data["Optimizer"].append(name)
                        results_data["Final X"].append(f"{final_x:.4f}")
                        results_data["Final Y"].append(f"{final_y:.4f}")
                        results_data["Final Loss"].append(f"{loss_fn(final_x, final_y):.4f}")
                        results_data["Steps"].append(iterations)  # Show iterations instead of len(trace)
                    
                    # Update the table placeholder with the actual values
                    results_table_placeholder.table(results_data)
                    
                    # Update the description placeholder
                    optimizer_desc_placeholder.markdown("""
                    ## Optimizer Descriptions
                    
                    - **SGD**: Basic gradient descent algorithm
                    
                    - **Momentum**: Adds fraction of previous update to the current one
                    
                    - **Nesterov**: Momentum variant that looks ahead to calculate gradient
                    
                    - **AdaGrad**: Adapts learning rate based on parameter history
                    
                    - **RMSProp**: Uses exponential moving average of squared gradients
                    
                    - **AdaDelta**: Improves RMSProp with parameter update history
                    
                    - **Adam**: Combines momentum and RMSProp concepts
                    """)
                

# Run the app
if __name__ == "__main__":
    main()