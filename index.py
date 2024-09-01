import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Константы для обработки
THRESHOLD = 21.5  
SMOOTHING_WINDOW = 4  

# Укажите путь к модели (относительный путь, если файл в главной директории репозитория)
MODEL_PATH = 'model.keras'  # Используйте относительный путь к модели
model_path = MODEL_PATH
# Загрузка модели Keras с использованием Streamlit caching
@st.cache_resource
def load_keras_model(model_path):
    try:
        model = load_model(model_path)
        st.success(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def load_audio_file(file_path, sr=44100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
    except Exception as e:
        st.write(f"Error with librosa: {e}. Trying scipy...")
        sample_rate, audio = wavfile.read(file_path)
        if sample_rate != sr:
            audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
    return audio, sample_rate

def extract_features(audio, sample_rate, frame_length, feature_type):
    samples_per_frame = int(sample_rate * frame_length)
    total_frames = int(len(audio) / samples_per_frame)
    features = []

    for i in range(total_frames):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame = audio[start_idx:end_idx]

        if len(frame) < samples_per_frame:
            frame = np.pad(frame, (0, samples_per_frame - len(frame)), 'constant')

        if feature_type == 'raw':
            feature = frame.reshape(-1, 1)
        features.append(feature)

    features = np.array(features)
    
    if feature_type == 'raw':
        max_length = max(len(f) for f in features)
        features = np.array([np.pad(f, ((0, max_length - len(f)), (0, 0)), 'constant') for f in features])
        features = np.expand_dims(features, -1)

    return features

def smooth_predictions(predictions, window_size):
    if window_size < 2:
        return predictions  
    return np.convolve(predictions, np.ones(window_size) / window_size, mode='valid')

def process_and_plot(file_path, model, samplerate_target, samplesize_ms):
    frame_length = samplesize_ms / 1000

    audio, sample_rate = load_audio_file(file_path, sr=samplerate_target)
    features = extract_features(audio, sample_rate, frame_length, feature_type='raw')

    # Преобразование features в формат, подходящий для модели Keras
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    predictions = model.predict(features_tensor).flatten()

    smoothed_predictions = smooth_predictions(predictions, SMOOTHING_WINDOW)
    filtered_predictions = smoothed_predictions[smoothed_predictions > THRESHOLD]
    average_value = np.mean(filtered_predictions) if len(filtered_predictions) > 0 else 0  

    time_axis_original = np.linspace(0, len(audio) / sample_rate, num=len(predictions))
    time_axis_smoothed = np.linspace(0, len(audio) / sample_rate, num=len(smoothed_predictions))

    fig, ax = plt.subplots()
    ax.plot(time_axis_original, predictions, label='Original Predicted Flow Rate L/min', alpha=0.5)
    ax.plot(time_axis_smoothed, smoothed_predictions, label='Smoothed Predicted Flow Rate L/min', linewidth=2)
    ax.axhline(y=average_value, color='r', linestyle='--', label=f'Average Flow Rate L/min (>{THRESHOLD})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flow Rate L/min')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f'Predictions for {base_name}')

    st.pyplot(fig)
    st.write(f'Predicted average for {base_name} with threshold {THRESHOLD} and smoothing window {SMOOTHING_WINDOW}: {average_value}')

def main():
    st.title("Audio Analysis with Deep Learning using Keras")

    # Загрузка модели
    model = load_keras_model(MODEL_PATH)

    if model:
        folder_path = st.text_input('Enter folder path with .wav files:')

        if folder_path:
            samplerate_target = st.number_input('Enter sample rate (default is 44100):', value=44100)
            samplesize_ms = st.number_input('Enter sample size in milliseconds (default is 50):', value=50)

            if st.button('Process Files'):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path) and file_path.lower().endswith('.wav'):
                        process_and_plot(file_path, model, samplerate_target, samplesize_ms)

if __name__ == "__main__":
    main()


