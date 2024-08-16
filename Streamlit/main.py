import streamlit as st

st.title('Vision Model Fine-Tuner')

with st.form(key='my_form'):
    st.write('Training Parameters')

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            label='Model',
            options=['ResNet50', 'ResNet18', 'InceptionV3'],
        )

        epochs = st.number_input(
            label='Number of Epochs',
            step=10,
            value=100,
        )

        batch_size = st.number_input(
            label='Batch Size',
            step=32,
            value=256,
        )

        learning_rate = st.number_input(
            label='Learning Rate',
            value=0.0001,
            step=0.0001,
            format='%.4f',
        )

        weight_decay = st.number_input(
            label='Weight Decay',
            value=0.0001,
            step=0.0001,
            format='%.4f',
        )

        log_interval = st.number_input(
            label='Log Interval',
            step=5,
            value=5,
        )

        num_gpus = st.number_input(
            label='Number of GPUs',
            value=1,
        )

    with col2:

        optimizer = st.radio(
            label='Optimizer',
            options=['ADAM', 'SGD'],
        )

        decay_type = st.radio(
            label='Learning Rate Decay',
            options=['cosine_warmup', 'step_warmup', 'step'],
        )

        use_mixed_precision = st.radio(
            label='Mixed Precision',
            options=['No', 'Yes'],
        )

    st.write('Data Loading')

    col3, col4 = st.columns(2)

    with col3:
        num_workers = st.number_input(
            label='Number of Workers',
            value=16,
        )

    with col4:
        seed = st.number_input(
            label='Seed',
            value=42,
        )

    st.write('Directory Paths')
    st.caption('If the data is not pre-split, only fill out the Training Directory')
    
    col5, col6 = st.columns(2)

    with col5:
        train_dir = st.text_input(
            label='Training Directory',
        )

        valid_dir = st.text_input(
            label='Validation Directory',
        )

        test_dir = st.text_input(
            label='Test Directory',
        )

    with col6:
        checkpoint_name = st.text_input(
            label='Checkpoint Name',
            value='baseline',
        )
        
        save_dir = st.text_input(
            label='Save Directory',
            value='checkpoints',
        )
    submit_button = st.form_submit_button(label='Submit')