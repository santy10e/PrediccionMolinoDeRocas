# Clasificador de Audio para Máquina de Bolas

Este proyecto consiste en un programa para la predicción de sonido de una máquina de bolas, desarrollado para el laboratorio de mecánica de rocas de la Universidad Nacional de Loja.

## Descripción

El programa utiliza técnicas de procesamiento de audio y aprendizaje automático para clasificar sonidos grabados de una máquina de bolas en dos categorías: "normal" y "anormal". Se emplea la biblioteca `librosa` para la extracción de características de audio y un clasificador de bosque aleatorio (`RandomForestClassifier`) para realizar las predicciones.

## Estructura del Proyecto

- `main.py`: Código principal del programa.
- `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
- `audio_classifier_model.pkl`: Archivo del modelo entrenado.
- `label_encoder.pkl`: Archivo del codificador de etiquetas.
- Carpeta `content`: Contiene los archivos de audio utilizados para el entrenamiento y la prueba.
- Carpeta `src`: Contiene imágenes utilizadas en la interfaz de usuario.

## Requisitos

- Python 3.x
- `librosa`
- `numpy`
- `matplotlib`
- `joblib`
- `soundfile`
- `streamlit`
- `scikit-learn`

## Instalación

1. Clona el repositorio:
   git clone [https://github.com/santy10e/tu_proyecto.git](https://github.com/santy10e/PrediccionMolinoDeRocas.git)

## Entrena el modelo:

En la interfaz de Streamlit, haz clic en el botón "Entrenar Modelo" para iniciar el proceso de entrenamiento.

## Cargar y clasificar un archivo de audio:
Carga un archivo de audio en formato .wav y el modelo predecirá si el sonido es "normal" o "anormal".
Se mostrará el resultado junto con una imagen indicativa (verde para normal y rojo para anormal).

## Reproducción de audios de muestra:
Puedes reproducir audios normales y anormales de muestra para comparar.

Nota: los audios de prueba se encuentran en la carpeta 'test'
