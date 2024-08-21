import streamlit as st
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import threading
import pandas as pd
import os
from train import train

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
# Function to update progress bar
def update_progress_bar(result_dict, epochs, placeholder):
    current_epoch = result_dict['epoch'][-1] if isinstance(result_dict['epoch'], list) and result_dict['epoch'] else len(result_dict['train_acc'])
    progress_text = f"Epoch: {current_epoch} / {(epochs)}"
    progress = (current_epoch) / epochs
    placeholder.progress(progress, text=progress_text)

# Function to plot live metrics with dual y-axes
def plot_live_metrics(result_dict, epochs, placeholder):
    fig, ax1 = plt.subplots()

    # Ensure lists are not empty and handle empty arrays
    if result_dict.get('train_acc'):
        epochs_progress = np.arange(1, len(result_dict['train_acc']) + 1)
        current_epoch = result_dict['epoch'][-1] if isinstance(result_dict['epoch'], list) and result_dict['epoch'] else len(result_dict['train_acc'])

        # Plot training and validation loss on primary y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ln1 = ax1.plot(epochs_progress, result_dict['train_loss'], color='tab:red', label='Train Loss')
        ln2 = ax1.plot(epochs_progress, result_dict['val_loss'], color='tab:red', linestyle='dashed', label='Validation Loss')

        # Plot training and validation accuracy on secondary y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ln3 = ax2.plot(epochs_progress, result_dict['train_acc'], color='tab:blue', label='Train Accuracy')
        ln4 = ax2.plot(epochs_progress, result_dict['val_acc'], color='tab:blue', linestyle='dashed', label='Validation Accuracy')

        # Combine legends from both y-axes
        lns = ln1 + ln2 + ln3 + ln4
        labels = [line.get_label() for line in lns]
        ax1.legend(lns[:2], labels[:2], loc='upper left')
        ax2.legend(lns[2:], labels[2:], loc='upper right')

        # Set title and layout
        ax1.set_title(f"Training Progress - Epoch {current_epoch}")
        ax1.grid()
        plt.tight_layout()

        # Save the plot as an image
        #plt.savefig('path/to/your/plot.png', bbox_inches='tight')  # Update the path accordingly
        #plt.close(fig)

        # Display the plot in Streamlit
        placeholder.pyplot(fig)
    
def update_table(result_dict, placeholder):

    if 'test_acc' in result_dict and isinstance(result_dict['test_acc'], list) and result_dict['test_acc']:
        test_acc = round(result_dict['test_acc'][-1], 2)
    else:
        test_acc = 0.00
        
    data = {
        'Best Model Epoch': [0],
        'Train Loss': [0],
        'Train Accuracy': [0],
        'Top 1 Accuracy': [0],
        'Test Accuracy': [test_acc],
    }
    df = pd.DataFrame(data)
    placeholder.dataframe(df)

# Streamlit UI setup
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

        amp = st.radio(
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
        pretrained_path = st.text_input(
            label='Pretrained Model Weights Path',
        )
        
        checkpoint_name = st.text_input(
            label='Checkpoint Name',
            value='baseline',
        )
        
        checkpoint_dir = st.text_input(
            label='Save Directory',
            value='checkpoints',
        )

    submit_button = st.form_submit_button(label='Submit')

# Placeholder for progress bar
progress_placeholder = st.progress(0, text="Epoch: 0 / 0")

# Placeholder for plot
plot_placeholder = st.empty()

fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Progress')
ax1.grid()
ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy')
plot_placeholder.pyplot(fig)

# Data table to track best model
data = {
        'Best Model Epoch': [0],
        'Train Loss': [0],
        'Train Accuracy': [0],
        'Top 1 Accuracy': [0],
        'Test Accuracy': [0],
}

df = pd.DataFrame(data)

# Display the DataFrame in Streamlit
table_placeholder = st.empty()
table_placeholder.dataframe(df)

# Handle submit
if submit_button:
    
    # Path to JSON file
    file_path = os.path.join(checkpoint_dir, checkpoint_name, 'result_dict.json')
    image_path = os.path.join(checkpoint_dir, checkpoint_name, 'learning_curve.png')

    def run_training():
        train(
            model,
            epochs,
            batch_size,
            learning_rate,
            weight_decay,
            log_interval,
            num_gpus,
            optimizer,
            decay_type,
            amp,
            num_workers,
            seed,
            train_dir,
            valid_dir,
            test_dir,
            pretrained_path,
            checkpoint_name,
            checkpoint_dir,
        )

    # Start training in a background thread
    training_thread = threading.Thread(target=run_training)
    training_thread.start()

    # Monitor and update plot
    while training_thread.is_alive():
        if os.path.exists(file_path):
            result_dict = read_json_file(file_path)
            update_progress_bar(result_dict, epochs, progress_placeholder)
            plot_live_metrics(result_dict, epochs, plot_placeholder)
            update_table(result_dict, table_placeholder)
        time.sleep(5)  # Refresh every 5 seconds

    # Ensure final plot update
    if os.path.exists(file_path):
        result_dict = read_json_file(file_path)
        update_progress_bar(result_dict, epochs, progress_placeholder)
        plot_live_metrics(result_dict, epochs, plot_placeholder)
        update_table(result_dict, table_placeholder)
    st.write("Training complete!")
