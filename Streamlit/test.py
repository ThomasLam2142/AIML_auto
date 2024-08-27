import streamlit as st

# Initialize session state to keep track of the current page
if "page" not in st.session_state:
    st.session_state.page = "Training"

# Sidebar navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Training", use_container_width=True):
    st.session_state.page = "Training"
if st.sidebar.button("Inference", use_container_width=True):
    st.session_state.page = "Inference"

# Training Page
if st.session_state.page == "Training":
    st.title("Training Page")
    st.write("This is the training page. You can start training your model here.")

# Inference Page
elif st.session_state.page == "Inference":
    st.title("Inference Page")
    st.write("This is the inference page. You can run inference on your model here.")
    # Add your inference-related code and UI elements here
