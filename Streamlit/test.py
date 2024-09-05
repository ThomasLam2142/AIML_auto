import streamlit as st
from PIL import Image

# Get image
image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Display the uploaded image
if image is not None:
    image = Image.open(image)
    st.image(image, caption="Input Image", use_column_width=True)