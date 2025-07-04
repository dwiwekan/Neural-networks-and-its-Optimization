import streamlit as st


def main():
    st.title("🧠 Neural Networks & Optimization: Practical Guide")

    st.markdown("""
        Welcome to the **Neural Networks and Optimization** interactive guide! 🚀
        
        This app is designed to teach you the fundamentals of Artificial Neural Networks (ANN) and Optimization techniques through interactive examples and visualizations.

        Here's what you can explore:
    """)

    st.header("📖 Sections")

    st.markdown("""
    - **Page 1: ANN Basics**: Learn about the fundamentals of Artificial Neural Networks and interactively visualize forward propagation with simple networks.
    - **Page 2: Gradient Descent**: Visualize gradient descent on a convex/non-convex function and understand how it converges with different learning rates.
    - **Page 3: Gradient Descent Variants**: Compare different gradient descent variants and see how they perform on complex optimization landscapes.
    - **Page 4: Challenges in Optimization**: Explore issues like local minima, saddle points, and cliffs, and how to deal with them effectively.
    """)

    st.image("https://miro.medium.com/v2/1*f9a162GhpMbiTVTAua_lLQ.png", caption="Gradient Descent Example", use_container_width=True)

    st.markdown("""
    **Interactive Learning**: Each section of the app provides interactive visualizations that update in real-time as you change parameters. This way, you can understand the impact of different choices on the behavior of a model or optimization algorithm.
    """)

    st.markdown("Happy Learning! 🚀")

if __name__ == "__main__":
    main()
