import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Загрузка модели TFLite
@st.cache_resource
def load_tflite_model():
    # Загрузка TFLite модели
    model_path = 'model_cheat.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Получение информации о тензорах входа и выхода модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Константы
THRESHOLD = 21.5  
SMOOTHING_WINDOW = 4  

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
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

def predict_with_tflite(interpreter, input_data):
    # Ensure input data is of the correct type (float32)
    input_data = np.array(input_data, dtype=np.float32)

    # Ensure the input data matches the expected input shape
    expected_shape = input_details[0]['shape']
    input_data = np.reshape(input_data, expected_shape)

    # Устанавливаем данные для входного тензора
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Выполняем предсказание
    interpreter.invoke()
    
    # Получаем результаты
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data.flatten()

def process_and_plot(file_path, interpreter, samplerate_target, samplesize_ms):
    frame_length = samplesize_ms / 1000

    audio, sample_rate = load_audio_file(file_path, sr=samplerate_target)
    features = extract_features(audio, sample_rate, frame_length, feature_type='raw')

    # Преобразование функций в нужный формат
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    predictions = predict_with_tflite(interpreter, features_tensor)

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
    st.title('Audio File Processing with TensorFlow Lite')

    # Загружаем файл через интерфейс Streamlit
    uploaded_file = st.file_uploader("Upload a .wav file", type="wav")
    
    # Проверяем, был ли загружен файл
    if uploaded_file is not None:
        # Сохраняем загруженный файл во временную директорию
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Выполняем анализ загруженного файла
        process_and_plot("temp_audio.wav", interpreter, samplerate_target=44100, samplesize_ms=50)
    else:
        st.write("Please upload a .wav file to analyze.")

if __name__ == '__main__':
    main()

