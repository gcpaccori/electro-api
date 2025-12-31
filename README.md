# Electro API: Sistema de Lectura de Medidores (v2.5)

![Electro Sur Este](https://img.shields.io/badge/Cliente-Electro%20Sur%20Este-0054a6)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/AI-YOLOv11%20Nano-purple)](https://docs.ultralytics.com/)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()

> **Electro API** es una solución de visión artificial de alto rendimiento diseñada para procesar masivamente (>700,000 imágenes/mes) fotografías de medidores de energía eléctrica. Optimizada para CPU y entornos productivos reales.
<img width="1905" height="908" alt="image" src="https://github.com/user-attachments/assets/9c89a92d-073d-4467-b2fe-3a4c188737b9" />
https://electro-api-gcpaccori.leapcell.app/
---

## 📋 Tabla de Contenidos

1. Arquitectura del Sistema  
2. Lógica Avanzada de Procesamiento  
3. Requisitos Previos  
4. Instalación Local  
5. Despliegue en Producción  
6. Documentación de la API  
7. Interfaz Web de Diagnóstico  
8. Estructura del Proyecto  
9. Solución de Problemas  

---

## 🏗 Arquitectura del Sistema

El sistema utiliza una arquitectura en cascada dividida en etapas especializadas:

1. **Detección de Display**  
   Modelo YOLOv11 Nano que localiza exclusivamente la pantalla del medidor.

2. **Transformación Geométrica**  
   Corrección automática de rotación, escala y perspectiva.

3. **Reconocimiento de Dígitos**  
   Segundo modelo YOLOv11 Nano para números y puntos decimales.

4. **Heurística de Negocio**  
   Reglas geométricas y validaciones finales.

---

## 🧠 Lógica Avanzada de Procesamiento

### A. Geometría Inteligente

| Tipo de Display | Ratio ancho/alto | Acción |
|----------------|------------------|--------|
| Vertical | < 0.85 | Rotación ±90° y selección por confianza |
| Cuadrado | 0.85 – 1.3 | Escalado proporcional |
| Horizontal | > 1.3 | Warping a 400x150 px |

### B. Filtro de Superposición (Custom NMS)

- Si dos dígitos están a menos de 20 px horizontalmente:
  - Se elimina el de menor confianza
- Los puntos decimales no se filtran

### C. Regla del Último Punto

- Si se detectan múltiples puntos:
  - Se conserva solo el más a la derecha

---

## 💻 Requisitos Previos

- Sistema Operativo: Windows, Linux, macOS  
- Python: 3.9 o superior (probado en 3.10)  
- Hardware: CPU (no requiere GPU)  

---

## ⚙️ Instalación Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/electro-api.git
cd electro-api
```

### 2. Crear Entorno Virtual

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar Modelos

Los siguientes archivos deben estar en la raíz del proyecto:

- `display_detection.pt`  
- `digit_recognition.pt`  

### 5. Ejecutar Aplicación

```bash
python app.py
```

Servidor disponible en:  
http://0.0.0.0:5000

---

## 🚀 Despliegue en Producción

### Render / Heroku (CPU)

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
gunicorn app:app --timeout 120
```

⚠️ El timeout es crítico para permitir la carga inicial de los modelos YOLO.

---

## 📡 Documentación de la API

### Endpoint

```http
POST /detect
```

### Parámetros (multipart/form-data)

| Campo | Requerido | Descripción |
|------|-----------|-------------|
| image | Sí | Imagen del medidor |
| include_visuals | No | "true" devuelve imágenes debug en Base64 |

---

### Escenario A: Procesamiento Masivo

```json
{
  "filename": "Suministro_293848.jpg",
  "display_detected": true,
  "reading": "14502.6"
}
```

---

### Escenario B: Auditoría Visual

```json
{
  "filename": "Suministro_293848.jpg",
  "display_detected": true,
  "reading": "14502.6",
  "debug_original": "BASE64...",
  "debug_warp": "BASE64..."
}
```

---

## 🖥 Interfaz Web de Diagnóstico

Disponible en `/`

- Drag & Drop múltiple  
- Cola secuencial de imágenes  
- Visualización de rotaciones y recortes  

---

## 📂 Estructura del Proyecto

```plaintext
electro-api/
├── app.py
├── requirements.txt
├── display_detection.pt
├── digit_recognition.pt
├── templates/
│   └── index.html
├── Procfile
└── README.md
```

---

## 🔧 Solución de Problemas

### Modelos no cargados

- Verificar archivos .pt  
- Revisar logs de inicio  

### Detecciones duplicadas

- Ajustar `min_dist` en `solve_overlapping_digits`  

### Timeout / Out of Memory

- Usar Gunicorn  
- Evitar imágenes excesivamente grandes  

---

**Desarrollado para Electro Sur Este S.A.A.**  
*Gerencia TIC*
