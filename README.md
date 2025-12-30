# Electro API: Lectura Inteligente de Medidores (YOLO11 Nano)

> API de visión artificial de alto rendimiento para la lectura automática de medidores de energía. Implementa una arquitectura en cascada con corrección de perspectiva, potenciada por **YOLO11 Nano**.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/flask-3.0-green)](https://flask.palletsprojects.com/)
[![AI Engine](https://img.shields.io/badge/YOLO-11%20Nano-purple)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-grey)]()

---

## 🖼️ Demostración del Pipeline

El sistema no simplemente "busca números". Sigue un proceso cognitivo similar al humano:
1. **Localiza** la pantalla del medidor.
2. **Corrige** la perspectiva (Warp) para aplanar la imagen.
3. **Lee** los dígitos secuencialmente sobre la imagen corregida.

![Pipeline de Detección](<img width="910" height="704" alt="image" src="https://github.com/user-attachments/assets/f015f4e5-f72e-423d-9a8f-87345a0efec3" />)

---

## 💡 ¿Por qué YOLO11 Nano?

Este proyecto ha sido migrado a **YOLO11 Nano**, la versión más reciente y ligera de la arquitectura YOLO.

* **Velocidad Extrema:** Optimizado para inferencia en tiempo real en CPUs.
* **Peso Pluma:** Los modelos pesan menos, lo que reduce el tiempo de arranque en servidores como **Render**.
* **Precisión/Costo:** Ofrece el mejor balance para tareas de detección de bordes y caracteres simples sin requerir GPUs costosas.

---

## ⚙️ Lógica de Procesamiento (The Cascade)

El backend (`app.py`) ejecuta la siguiente lógica estricta para garantizar la calidad de la lectura:

1.  **Detección de Pantalla (Display Model):**
    * Escanea la foto completa.
    * Extrae la ROI (Región de Interés) del display LCD/LED.
2.  **Pre-procesamiento Geométrico (Warping):**
    * Recorta el display detectado.
    * Realiza un `resize` forzado a **400x150px**. Esto normaliza el tamaño de los dígitos independientemente de la distancia de la foto.
3.  **Reconocimiento de Dígitos (Digit Model):**
    * Se ejecuta **solo** sobre la imagen recortada y normalizada.
    * Usa `iou=0.3` para filtrar detecciones fantasma o superpuestas.
4.  **Algoritmo de Lectura:**
    * Ordena las coordenadas X de los dígitos detectados.
    * Reconstruye el valor numérico final (ej: `12345.6`).

---

## 🚀 Despliegue en Render (CPU Only)

Este proyecto está configurado nativamente para funcionar en el **Free Tier de Render** (que no tiene GPU).

1.  **Nuevo Web Service:** Conecta tu repo de GitHub.
2.  **Runtime:** Python 3.
3.  **Build Command:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Start Command:**
    ```bash
    gunicorn app:app --timeout 120
    ```
    *(El timeout de 120s es vital para permitir la carga de los modelos YOLO11 en memoria la primera vez).*

---

## 📦 Instalación Local

Si deseas correrlo en tu PC (con o sin GPU):

1.  **Clonar:**
    ```bash
    git clone [https://github.com/TU_USUARIO/electro-api.git](https://github.com/TU_USUARIO/electro-api.git)
    cd electro-api
    ```

2.  **Instalar Dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar:**
    ```bash
    python app.py
    ```
    Visita `http://localhost:5000` para ver la interfaz de Electro Sur Este.

---

## 📡 Endpoints de la API

### `POST /detect`
Envía una imagen para procesar.

* **Body (Multipart/Form-Data):**
    * `image`: Archivo de imagen (jpg, png).

* **Respuesta (JSON):**
    ```json
    {
      "success": true,
      "reading": "1045.2",
      "debug_original": "...base64_string...",
      "debug_warp": "...base64_string..."
    }
    ```

---

## 📂 Estructura del Proyecto
