
# Electro API: Detección de Lecturas en Cascada con YOLOv11n

> Una API robusta basada en Flask para la lectura automática de medidores digitales, utilizando un enfoque de visión artificial de dos etapas.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/flask-3.0-green)](https://flask.palletsprojects.com/)
[![ML Engine](https://img.shields.io/badge/YOLO-v11-purple)](https://docs.ultralytics.com/)

---

## 🖼️ Demostración

A continuación se muestra un ejemplo del procesamiento de la API: primero detecta la pantalla (cuadro verde) y luego busca los dígitos únicamente dentro de esa área (cuadros rojos).

![Demo del funcionamiento de Electro API]<img width="910" height="704" alt="image" src="https://github.com/user-attachments/assets/f015f4e5-f72e-423d-9a8f-87345a0efec3" />

---

## 💡 Descripción del Proyecto

Este proyecto implementa una API RESTful diseñada para extraer lecturas numéricas de imágenes de medidores eléctricos u otros dispositivos con pantallas digitales.

A diferencia de los enfoques tradicionales que buscan todo a la vez, esta API utiliza una **lógica secuencial (en cascada)** inteligente para mejorar la precisión y reducir falsos positivos:

1.  **Etapa 1 - Detección de Pantalla:** Un modelo YOLOv11n especializado (`display_detection.pt`) analiza la imagen completa para localizar el área de la pantalla LCD/LED.
2.  **Etapa 2 - Recorte y Reconocimiento de Dígitos:** Si se encuentra una pantalla, la imagen se recorta automáticamente a esa área de interés. Un segundo modelo YOLOv11n (`digit_recognition.pt`) busca los dígitos numéricos solo dentro de ese recorte.

Esta metodología asegura que el modelo de dígitos no se confunda con números o textos irrelevantes fuera de la pantalla del dispositivo.

---

## 🚀 Características Principales

* **Arquitectura de Dos Etapas:** Mayor precisión al enfocar la detección de dígitos solo en áreas relevantes.
* **Optimizado para CPU:** Configurado explícitamente para funcionar en entornos sin GPU (como el plan gratuito de Render), evitando conflictos de drivers CUDA.
* **Respuesta Rica:** El endpoint devuelve un JSON con los datos detectados y una versión en Base64 de la imagen procesada con las detecciones dibujadas.
* **Interfaz Web Básica:** Incluye una plantilla HTML simple en la ruta raíz `/` para pruebas rápidas.
* **Lista para Producción:** Configurada para usar Gunicorn como servidor WSGI en despliegues.

---

## 🛠️ Stack Tecnológico

* **Python 3.x**
* **Flask:** Framework web ligero para la API.
* **Ultralytics YOLOv11:** Motor de detección de objetos de última generación.
* **OpenCV & Pillow (PIL):** Para manipulación y procesamiento de imágenes.
* **Gunicorn:** Servidor HTTP WSGI para producción.

---

## 📦 Instalación y Uso Local

### Prerrequisitos
* Python instalado.
* Tener los archivos de modelo `display_detection.pt` y `digit_recognition.pt` en la raíz del proyecto.

### Pasos

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/electro-api.git](https://github.com/TU_USUARIO/electro-api.git)
    cd electro-api
    ```

2.  **Crear y activar entorno virtual (Recomendado):**
    ```bash
    # En Windows
    python -m venv venv
    .\venv\Scripts\activate

    # En macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el servidor de desarrollo:**
    ```bash
    python app.py
    ```
    La API estará disponible en `http://localhost:5000`.

---

## 📡 Documentación de la API

### Endpoint: `/detect`

* **Método:** `POST`
* **Descripción:** Procesa una imagen cargada y devuelve las detecciones.
* **Body (form-data):**
    * `image`: (Archivo, requerido) La imagen del medidor a analizar.

#### Ejemplo de Respuesta Exitosa (JSON):

```json
{
  "success": true,
  "detections": [
    {
      "box": [ 450, 210, 485, 310 ],
      "confidence": "0.92",
      "label": "1"
    },
    {
      "box": [ 490, 212, 530, 308 ],
      "confidence": "0.89",
      "label": "2"
    }
    // ... más dígitos
  ],
  "processed_image": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgI..."
}
