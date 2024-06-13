import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import joblib
import soundfile as sf
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Función para extraer características de los archivos de audio
def extract_features(sound, sr):
    mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    chroma = librosa.feature.chroma_stft(y=sound, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)
    mel = librosa.feature.melspectrogram(y=sound, sr=sr)
    mel_mean = np.mean(mel.T, axis=0)
    contrast = librosa.feature.spectral_contrast(y=sound, sr=sr)
    contrast_mean = np.mean(contrast.T, axis=0)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sr)
    tonnetz_mean = np.mean(tonnetz.T, axis=0)
    return np.hstack([mfccs_mean, chroma_mean, mel_mean, contrast_mean, tonnetz_mean])

# Función para cargar un archivo de audio desde la interfaz de Streamlit
def load_audio(file):
    if file is not None:
        audio_data, sample_rate = sf.read(file)
        return audio_data, sample_rate
    else:
        return None, None

def train_model():
    # Ruta donde se encuentran los datos de entrenamiento
    data_path = '../content/train_audio'  # Ajustar según sea necesario
    
    # Cargar los datos de entrenamiento y etiquetas
    X_train = []
    y_train = []

    status_text = st.empty()
    status_text.text(f"Buscando archivos en: {data_path}")

    process_status = st.empty()  # Contenedor para actualizar el estado del procesamiento

    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                status_text.text(f"Procesando archivo: {file_path}")
                try:
                    audio_data, sample_rate = sf.read(file_path)
                    if audio_data is None or len(audio_data) == 0:
                        st.error(f"Error al leer el archivo: {file_path}")
                        continue
                    features = extract_features(audio_data, sample_rate)
                    X_train.append(features)
                    if 'anormal' in file.lower() or 'abnormal' in file.lower():
                        y_train.append('anormal')
                        process_status.text(f"Procesando archivo: {file_path} | Etiqueta asignada: anormal")
                    elif 'normal' in file.lower():
                        y_train.append('normal')
                        process_status.text(f"Procesando archivo: {file_path} | Etiqueta asignada: normal")
                    else:
                        st.warning(f"No se pudo determinar la etiqueta para el archivo: {file_path}")
                except Exception as e:
                    st.error(f"Error procesando el archivo {file_path}: {e}")
                    continue

    # Convertir a arrays numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Verificar formas y contenido
    st.write(f"Shape of X_train: {X_train.shape}")
    st.write(f"Shape of y_train: {y_train.shape}")
    assert X_train.size > 0, "X_train está vacío"
    assert y_train.size > 0, "y_train está vacío"
    assert X_train.ndim == 2, f"X_train no es 2D: {X_train.ndim} dimensiones"
    assert y_train.ndim == 1, f"y_train no es 1D: {y_train.ndim} dimensiones"

    # Verificar la distribución de las clases
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    # st.write(f"Class distribution: {class_distribution}")

    # Verificar si hay al menos dos clases presentes
    if len(unique) < 2:
        st.error("El conjunto de datos debe contener al menos dos clases ('normal' y 'anormal').")
        return

    # Codificar etiquetas
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # Dividir los datos en conjuntos de entrenamiento y prueba de manera estratificada
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    # Verificar la distribución de las clases después de la división
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    class_distribution_train = dict(zip(unique_train, counts_train))
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    class_distribution_test = dict(zip(unique_test, counts_test))
    
    # st.write(f"Training set class distribution: {class_distribution_train}")
    # st.write(f"Test set class distribution: {class_distribution_test}")

    # Entrenar un clasificador (usando RandomForest como ejemplo)
    clf = RandomForestClassifier(random_state=42)
    
    # Búsqueda de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Mejor modelo
    best_clf = grid_search.best_estimator_
    # st.write(f"Best parameters found: {grid_search.best_params_}")

    # Validación cruzada
    scores = cross_val_score(best_clf, X_train, y_train, cv=5)
    # st.write(f"Cross-validation scores: {scores}")
    # st.write(f"Mean cross-validation score: {np.mean(scores)}")

    # Guardar el modelo entrenado y el LabelEncoder
    model_file = 'audio_classifier_model.pkl'
    label_encoder_file = 'label_encoder.pkl'
    joblib.dump(best_clf, model_file)
    joblib.dump(le, label_encoder_file)
    st.success("Modelo entrenado y guardado como: " + model_file)
    status_text.text("Entrenamiento completado")

# Cargar el modelo entrenado y el LabelEncoder
def load_model():
    model_file = 'audio_classifier_model.pkl'
    label_encoder_file = 'label_encoder.pkl'
    if os.path.exists(model_file) and os.path.exists(label_encoder_file):
        clf_loaded = joblib.load(model_file)
        le_loaded = joblib.load(label_encoder_file)
        return clf_loaded, le_loaded
    else:
        st.error("El modelo no se encuentra entrenado. Entrena el modelo primero.")
        return None, None

# Función para mostrar el gráfico de dominio del tiempo
def plot_waveform(audio_data, sample_rate):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(np.linspace(0, len(audio_data) / sample_rate, num=len(audio_data)), audio_data, color='b')
    ax.set_title("Audio en el dominio del tiempo")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud")
    ax.grid()
    return fig

# Interfaz de Streamlit
st.title("Clasificador de Audio")

# Botón para entrenar el modelo
if st.button("Entrenar Modelo"):
    with st.spinner('Entrenando modelo...'):
        train_model()

# Selección de archivo de audio para prueba
audio_file = st.file_uploader("Cargar archivo de audio", type=["wav"])

# Predicción con el modelo cargado
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    audio_data, sample_rate = load_audio(audio_file)
    
    if audio_data is not None:
        
        # Extraer características del audio
        audio_features = extract_features(audio_data, sample_rate)
        
        # Cargar el modelo entrenado
        clf_loaded, le_loaded = load_model()
        
        if clf_loaded is not None and le_loaded is not None:
            # Realizar la predicción
            audio_features = audio_features.reshape(1, -1)
            prediction = clf_loaded.predict(audio_features)
            pred_text = 'normal' if prediction[0] == le_loaded.transform(['normal'])[0] else 'anormal'

            # Mostrar el resultado con imágenes de semáforo
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            with col3:
                if pred_text == 'normal':
                    st.image('../src/verde3.jpg', caption='Predicción: NORMAL', width=150)
                else:
                    st.image('../src/rojo3.jpg', caption='Predicción: ANORMAL', width=150)
        
        # Mostrar gráfico de dominio del tiempo
        fig = plot_waveform(audio_data, sample_rate)
        st.pyplot(fig)

else:
    st.write("No se ha cargado ningún archivo de audio.")
    
# Reproducción de audios normales y anormales
st.subheader("Reproducir audios de muestra")
if st.button("Reproducir audio normal"):
    audio_normal_file = '../content/train_audio/section_00_source_train_normal_0016.wav'
    st.audio(audio_normal_file, format='audio/wav')

if st.button("Reproducir audio anormal"):
    audio_anormal_file = '../content/train_audio/section_00_source_train_anormal_0000.wav'
    st.audio(audio_anormal_file, format='audio/wav')
