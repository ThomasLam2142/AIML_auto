from transformers import Wav2Vec2Processor, HubertForCTC
import torch
import torchaudio

# Load Audio
def load_audio(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
    return waveform, target_sample_rate

# Preprocess Audio
def preprocess_audio(waveform):
    if waveform.shape[0] > 1:  # Check if the audio has more than one channel (stereo)
        waveform = waveform.mean(dim=0, keepdim=True)  # Average to get mono
    return waveform

# Feature Extraction and Inference
def transcribe_audio(model, processor, waveform, sample_rate=16000):
    # Convert waveform to input features for the model
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values
    input_values = input_values.squeeze(1)  # Ensure correct input dimensions
    
    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode the output logits to get transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

def main(input_audio_path):
    # Initialize HuBERT model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    # Load and preprocess audio
    waveform, sample_rate = load_audio(input_audio_path)
    waveform = preprocess_audio(waveform)

    # Transcribe and output text
    transcription = transcribe_audio(model, processor, waveform, sample_rate)
    print("Transcription:", transcription)

input_audio_path = "sample_audio.mp3"
main(input_audio_path)
