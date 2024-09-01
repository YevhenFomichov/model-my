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
    # Путь к вашей модели TensorFlow Lite
    model_path = 'model_cheat.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Получение информации о тензорах входа и выхода модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Порог и окно сглаживания
THRESHOLD = 21.5  
SMOOTHING_WINDOW = 4  

def load_audio_file(file_path, sr=44100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
    except Exception as e:
        st.write(f"Ошибка при загрузке с помощью librosa: {e}. Пробуем использовать scipy...")
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

def predict_with_tflite(interpreter, features):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Преобразуем признаки в нужный тип и форму
    input_data = np.array(features, dtype=np.float32)

    # Проверяем количество фрагментов и делим на батчи
    total_samples = input_data.shape[0]
    batch_size = input_details[0]['shape'][0]  # Ожидаемый размер батча (32)
    input_batches = []

    for i in range(0, total_samples, batch_size):
        batch = input_data[i:i + batch_size]
        
        # Если батч меньше, чем ожидается, добавляем нулевые значения для заполнения
        if batch.shape[0] < batch_size:
            # Создаем пустой массив правильной формы для заполнения
            padding = np.zeros((batch_size - batch.shape[0], input_data.shape[1], input_data.shape[2]), dtype=np.float32)
            
            # Проверяем форму перед объединением
            st.write(f"Форма батча: {batch.shape}, Форма заполнения: {padding.shape}")
            
            # Используем np.concatenate для объединения данных
            batch = np.concatenate((batch, padding), axis=0)
        
        input_batches.append(batch)

    predictions = []

    for batch in input_batches:
        # Устанавливаем данные для входного тензора
        interpreter.set_tensor(input_details[0]['index'], batch)

        # Выполняем предсказание
        interpreter.invoke()

        # Получаем результаты
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.extend(output_data.flatten())

    return np.array(predictions)

def process_and_plot(file_path, interpreter, samplerate_target, samplesize_ms):
    frame_length = samplesize_ms / 1000

    audio, sample_rate = load_audio_file(file_path, sr=samplerate_target)
    features = extract_features(audio, sample_rate, frame_length, feature_type='raw')

    # Прогнозирование с использованием TFLite модели
    predictions = predict_with_tflite(interpreter, features)

    if predictions.size == 0:
        st.error("Не удалось получить предсказания из модели. Проверьте логи для получения дополнительных сведений.")
        return

    smoothed_predictions = smooth_predictions(predictions, SMOOTHING_WINDOW)
    
    filtered_predictions = smoothed_predictions[smoothed_predictions > THRESHOLD]
    average_value = np.mean(filtered_predictions) if len(filtered_predictions) > 0 else 0  

    time_axis_original = np.linspace(0, len(audio) / sample_rate, num=len(predictions))
    time_axis_smoothed = np.linspace(0, len(audio) / sample_rate, num=len(smoothed_predictions))

    fig, ax = plt.subplots()
    ax.plot(time_axis_original, predictions, label='Оригинальный прогноз потока (л/мин)', alpha=0.5)
    ax.plot(time_axis_smoothed, smoothed_predictions, label='Сглаженный прогноз потока (л/мин)', linewidth=2)
    ax.axhline(y=average_value, color='r', linestyle='--', label=f'Средний поток (л/мин) (>{THRESHOLD})')
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Поток (л/мин)')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f'Прогнозы для {base_name}')
    st.pyplot(fig)

    st.write(f'Средний прогноз для {base_name} с порогом {THRESHOLD} и окном сглаживания {SMOOTHING_WINDOW}: {average_value}')

def main():
    st.title('Обработка аудиофайлов с использованием TensorFlow Lite')

    uploaded_file = st.file_uploader("Загрузите файл .wav", type="wav")
    
    if uploaded_file is not None:
        # Сохранение загруженного файла временно
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Анализ загруженного файла
        process_and_plot("temp_audio.wav", interpreter, samplerate_target=44100, samplesize_ms=50)
    else:
        st.write("Пожалуйста, загрузите файл .wav для анализа.")

if __name__ == '__main__':
    main()
