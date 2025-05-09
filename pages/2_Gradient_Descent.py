import os
os.environ['STREAMLIT_SERVER_WATCH_EXCLUSIONS'] = 'torch'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="Gradient Descent Visualization")

# Title and introduction
st.title("Gradient Descent: Interactive Visualization")
st.markdown("""
Gradient descent is an optimization algorithm used to minimize a function by iteratively moving 
in the direction of steepest descent. This app demonstrates how gradient descent works on different 
cost functions, allowing you to experiment with parameters like learning rate and starting position.
""")

# Function to create different cost functions
def get_cost_function(function_type):
    if function_type == "Parabola (Convex)":
        def cost_func(x, y):
            return x**2 + y**2
        
        def gradient(x, y):
            return np.array([2*x, 2*y])
        
        x_range = (-5, 5)
        y_range = (-5, 5)
        global_min = (0, 0)
        
    elif function_type == "Rosenbrock Function":
        def cost_func(x, y):
            a = 1
            b = 100
            return (a - x)**2 + b * (y - x**2)**2
        
        def gradient(x, y):
            a = 1
            b = 100
            dx = -2*(a - x) - 4*b*x*(y - x**2)
            dy = 2*b*(y - x**2)
            return np.array([dx, dy])
        
        x_range = (-2, 2)
        y_range = (-1, 3)
        global_min = (1, 1)
        
    elif function_type == "Himmelblau's Function":
        def cost_func(x, y):
            return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        
        def gradient(x, y):
            dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
            dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
            return np.array([dx, dy])
        
        x_range = (-5, 5)
        y_range = (-5, 5)
        global_min = (3, 2)  # One of several minima
    
    else:  # Default to a simple bowl-shaped function
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
        for path, label in paths:
            if len(path) > 0:
                path_x, path_y = zip(*path)
                path_z = [cost_func(x, y) for x, y in zip(path_x, path_y)]
                ax.plot(path_x, path_y, path_z, 'r-', linewidth=2.5, label=label)
                ax.plot([path_x[-1]], [path_y[-1]], [path_z[-1]], 'ro', markersize=7)
    
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
        for path, label in paths:
            if len(path) > 0:
                path_x, path_y = zip(*path)
                ax.plot(path_x, path_y, '-', linewidth=2.5, label=label)
                ax.plot([path_x[-1]], [path_y[-1]], 'o', markersize=7)
        
        # Add legend
        ax.legend()
    
    return ax

# Perform gradient descent
def gradient_descent(start_x, start_y, gradient_func, learning_rate, iterations, momentum=0):
    path = [(start_x, start_y)]
    x, y = start_x, start_y
    
    # For momentum calculation
    prev_step_x, prev_step_y = 0, 0
    
    for _ in range(iterations):
        # Calculate gradient
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

# Main function
def main():
    # Create sidebar for parameters
    st.sidebar.header("Gradient Descent Parameters")
    
    # Select cost function
    function_type = st.sidebar.selectbox(
        "Select Cost Function",
        ["Parabola (Convex)", "Rosenbrock Function", "Himmelblau's Function"]
    )
    
    # Get the selected cost function
    cost_func, gradient_func, x_range, y_range, global_min = get_cost_function(function_type)
    
    # Starting position
    # Starting position
    st.sidebar.subheader("Starting Position")
    start_x = st.sidebar.slider("X Starting Point", 
                            float(x_range[0]), 
                            float(x_range[1]), 
                            float(x_range[0] + 0.75 * (x_range[1] - x_range[0])),
                            step=0.01)  # Explicitly add step with float type

    start_y = st.sidebar.slider("Y Starting Point", 
                            float(y_range[0]), 
                            float(y_range[1]), 
                            float(y_range[0] + 0.75 * (y_range[1] - y_range[0])),
                            step=0.01)  # Explicitly add step with float type
    
    # Optimization parameters
    st.sidebar.subheader("Optimization Parameters")
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.5, 0.1, step=0.001)
    momentum = st.sidebar.slider("Momentum", 0.0, 0.99, 0.0, step=0.01)
    iterations = st.sidebar.slider("Maximum Iterations", 5, 500, 50)
    
    # Show global minimum
    st.sidebar.subheader("Function Information")
    st.sidebar.write(f"Global Minimum: {global_min}")

    # Create a placeholder for displaying the plots
    plot_placeholder = st.empty()
    
    # Create a two-column layout for the plots
    col1, col2 = st.columns(2)
    
    # Interactive visualizations for static view
    with col1:
        st.subheader("3D Surface View")
        fig1 = plt.figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111, projection='3d')
        plot_cost_function(cost_func, x_range, y_range, ax=ax1)
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Contour View")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_contour(cost_func, x_range, y_range, ax=ax2)
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Button to start animation
    if st.button("Start Gradient Descent"):
        with st.spinner("Running gradient descent..."):
            # Perform gradient descent
            path = gradient_descent(
                start_x, start_y, gradient_func, 
                learning_rate, iterations, momentum
            )
            
            # Also run with different learning rates for comparison
            path_faster = gradient_descent(
                start_x, start_y, gradient_func, 
                learning_rate * 2, iterations, momentum
            )
            
            path_slower = gradient_descent(
                start_x, start_y, gradient_func, 
                learning_rate * 0.5, iterations, momentum
            )
            
            # Create animation
            progress_placeholder = st.progress(0)
            anim_placeholder = st.empty()
            
            # For cost vs. iterations plot
            costs = []
            iterations_list = []
            
            # Animate the gradient descent process
            for i in range(min(50, len(path))):
                # Update progress bar
                progress_placeholder.progress((i + 1) / min(50, len(path)))
                
                # Calculate subset of path for this frame
                current_path = path[:i+1]
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Plot contour with current path
                plot_contour(cost_func, x_range, y_range, 
                             paths=[(current_path, f"LR={learning_rate}")], 
                             ax=ax1)
                
                # Track costs
                if len(current_path) > 0:
                    x, y = current_path[-1]
                    costs.append(cost_func(x, y))
                    iterations_list.append(i)
                
                # Plot cost vs iterations
                ax2.plot(iterations_list, costs, 'b-')
                ax2.set_xlabel('Iterations')
                ax2.set_ylabel('Cost')
                ax2.set_title('Cost vs Iterations')
                ax2.grid(True)
                
                # Display the figure
                anim_placeholder.pyplot(fig)
                plt.close(fig)
                
                # Add small delay
                time.sleep(0.1)
            
            # Show final state with comparison
            st.subheader("Learning Rate Comparison")
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_contour(cost_func, x_range, y_range, 
                        paths=[
                            (path, f"LR={learning_rate}"),
                            (path_faster, f"LR={learning_rate*2}"),
                            (path_slower, f"LR={learning_rate*0.5}")
                        ],
                        ax=ax)
            st.pyplot(fig)
            plt.close(fig)
            
            # Final cost
            if len(path) > 0:
                final_x, final_y = path[-1]
                final_cost = cost_func(final_x, final_y)
                st.success(f"Optimization Complete! Final position: ({final_x:.4f}, {final_y:.4f}), Final cost: {final_cost:.6f}")

# Run the app
if __name__ == "__main__":
    main()