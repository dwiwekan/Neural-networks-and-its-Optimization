import streamlit as st

# Title of your app
st.title('Hello, Streamlit! 🎈')

# Header
st.header('Welcome to your first Streamlit App')

# Simple text
st.write('This is an example of a Streamlit web application.')

# Interactive widgets
name = st.text_input('Enter your name:', '')
if name:
    st.write(f'Hello, {name}! 👋')

# Slider example
age = st.slider('Select your age:', 0, 100, 25)
st.write(f'You are {age} years old.')
