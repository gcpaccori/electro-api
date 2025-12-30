# Electro API: Sistema Industrial de Lectura de Medidores (v2.5)

![Electro Sur Este](https://img.shields.io/badge/Cliente-Electro%20Sur%20Este-0054a6)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/AI-YOLOv11%20Nano-purple)](https://docs.ultralytics.com/)
[![Flask](https://img.shields.io/badge/Backend-Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green)]()

> **Electro API** es una solución de visión artificial de alto rendimiento diseñada para procesar masivamente (>700,000 imágenes/mes) fotografías de medidores de energía. Utiliza una arquitectura en cascada (Cascade R-CNN style) optimizada para CPU, capaz de rectificar perspectivas y filtrar lecturas erróneas mediante lógica geométrica avanzada.
<img width="1905" height="908" alt="image" src="https://github.com/user-attachments/assets/9b7b7553-4848-48db-bd5d-6dd435d091c6" />

---

## 📋 Tabla de Contenidos
1. [Arquitectura del Sistema](#-arquitectura-del-sistema)
2. [Lógica Avanzada de Procesamiento](#-lógica-avanzada-de-procesamiento)
3. [Requisitos Previos](#-requisitos-previos)
4. [Instalación Local (Paso a Paso)](#-instalación-local-paso-a-paso)
5. [Despliegue en Producción (Render/Docker)](#-despliegue-en-producción)
6. [Documentación de la API](#-documentación-de-la-api)
7. [Interfaz Web de Diagnóstico](#-interfaz-web-de-diagnóstico)
8. [Estructura del Proyecto](#-estructura-del-proyecto)
9. [Solución de Problemas](#-solución-de-problemas)

---

## 🏗 Arquitectura del Sistema

El sistema no utiliza un enfoque monolítico. Divide el problema cognitivo en dos etapas especializadas para maximizar la precisión:

1.  **Etapa 1 (Display Detection):** Un modelo YOLOv11 Nano escanea la imagen completa (sin importar el fondo, cables o paredes) y localiza exclusivamente la pantalla LCD/LED del medidor.
2.  **Etapa 2 (Geometric Transformation):** El recorte detectado es analizado y transformado (rotado o redimensionado) según su factor de forma.
3.  **Etapa 3 (Digit Recognition):** Un segundo modelo YOLOv11 Nano extrae los dígitos y puntos decimales sobre la imagen ya normalizada.
4.  **Etapa 4 (Heurística de Negocio):** Se aplican filtros de superposición (NMS personalizado) y reglas de negocio (validación de puntos) para ensamblar la lectura final.

---

## 🧠 Lógica Avanzada de Procesamiento

Esta versión (v2.5) implementa algoritmos correctivos para situaciones de campo reales:

### A. Geometría Inteligente (Auto-Rotation)
El sistema calcula el `ratio = ancho / alto` del display detectado para decidir cómo tratarlo:

| Tipo de Display | Ratio Detectado | Acción del Algoritmo |
| :--- | :--- | :--- |
| **Vertical** | `< 0.85` | El medidor está girado. El sistema ejecuta **dos simulaciones simultáneas**: rota la imagen 90° a la izquierda y 90° a la derecha. Se queda con la lectura que genere mayor confianza acumulada. |
| **Cuadrado** | `0.85 - 1.3` | **Redimensionado Proporcional.** No se estira la imagen a un rectángulo ancho (lo que aplastaría los números). Se escala manteniendo su forma para preservar la legibilidad. |
| **Horizontal** | `> 1.3` | **Warping Estándar.** Se estira la imagen a `400x150px` para maximizar la separación entre dígitos. |

### B. Filtro de Superposición Manual (Custom NMS)
Resuelve el error común donde se detectan dos números en el mismo espacio (ej: un `0` y un `7` superpuestos).
* **Lógica:** Si dos dígitos detectados tienen sus centros horizontales a menos de **20 píxeles** de distancia, se elimina el de menor confianza.
* **Excepción Crítica:** Este filtro **ignora los puntos decimales**. Un punto puede estar pegado a un número sin ser eliminado.

### C. Regla del Último Punto
En medidores sucios, a veces se detectan manchas como puntos (ej: `1.4.5.2`).
* **Lógica:** Si se detectan múltiples puntos, el sistema elimina todos excepto el último (el situado más a la derecha), garantizando una lectura decimal válida.

---

## 💻 Requisitos Previos

* **Sistema Operativo:** Windows, macOS, o Linux (Ubuntu/Debian recomendado para producción).
* **Python:** Versión 3.9 o superior (probado en 3.10).
* **Hardware:** No requiere GPU. Optimizado para inferencia rápida en CPU.

---

## ⚙️ Instalación Local (Paso a Paso)

Sigue estos pasos para levantar el entorno de desarrollo en tu máquina.

### 1. Clonar el Repositorio
```bash
git clone [https://github.com/TU_USUARIO/electro-api.git](https://github.com/TU_USUARIO/electro-api.git)
cd electro-api
### 2. Crear Entorno Virtual (Recomendado)Aísla las librerías para evitar conflictos.En Windows:Bashpython -m venv venv
.\venv\Scripts\activate
En Linux/macOS:Bashpython3 -m venv venv
source venv/bin/activate
### 3. Instalar DependenciasInstala las librerías optimizadas (Torch CPU, Flask, Ultralytics).Bashpip install -r requirements.txt
### 4. Verificar ModelosAsegúrate de que los archivos de pesos (.pt) estén en la raíz del proyecto:display_detection.ptdigit_recognition.pt5. Ejecutar la AplicaciónBashpython app.py
Verás un mensaje indicando que el servidor corre en http://0.0.0.0:5000. Abre esa URL en tu navegador para ver la interfaz de prueba.🚀 Despliegue en ProducciónEsta API está lista para plataformas PaaS como Render.com o Heroku.Configuración para Render (Free Tier CPU)Al crear un "Web Service" en Render, usa esta configuración exacta:Environment: Python 3Build Command:Bashpip install -r requirements.txt
Start Command:Bashgunicorn app:app --timeout 120
### Nota: El flag --timeout 120 es crítico. La primera vez que arranca, YOLO descarga assets y carga modelos en memoria, lo que puede tomar más de los 30s por defecto.Variables de Entorno: No son necesarias (el código fuerza device='cpu' internamente).📡 Documentación de la APILa API tiene un único endpoint inteligente /detect que cambia su respuesta según si es consumido por un humano (Web) o un script masivo (Batch).POST /detectParámetros (Multipart/Form-Data)CampoRequeridoDescripciónimage✅ SíArchivo de imagen (JPG, PNG, BMP, WEBP).include_visuals❌ NoSi se envía como 'true', la respuesta incluirá las imágenes procesadas en Base64. Si se omite, devuelve JSON ligero.Escenario A: Procesamiento Masivo (Batch)Uso ideal: Scripts que procesan 700k imágenes. Respuesta ultraligera (<1KB).Request:Solo enviar el archivo image.Response (JSON):JSON{
  "filename": "Suministro_293848.jpg",
  "display_detected": true,
  "reading": "14502.6"
}
### Si no detecta display: "display_detected": false, "reading": "desactivado".Escenario B: Auditoría Visual (Web/Debug)Uso ideal: Verificar por qué falló una lectura específica.Request:Enviar image y include_visuals='true'.Response (JSON):JSON{
  "filename": "Suministro_293848.jpg",
  "display_detected": true,
  "reading": "14502.6",
  "debug_original": "/9j/4AAQSkZJRgABAQ...",  // Imagen original con cuadro verde (Base64)
  "debug_warp": "/9j/4AAQSkZJRgABAQ..."      // Recorte rectificado con números rojos (Base64)
}

### 🔧 Solución de Problemas1. La API devuelve "Error: Modelos no cargados"Verifica que los archivos .pt estén en la misma carpeta que app.py.Revisa los logs de la consola al iniciar. Deberías ver ✅ Modelos listos..2. Detecta números dobles (ej: 0 y 7 juntos)Esto está mitigado por la función solve_overlapping_digits en app.py. Si persiste, intenta aumentar el parámetro min_dist=20 en la línea 60 de app.py.3. Render da error "Out of Memory" o "Timeout"Asegúrate de usar el comando de inicio con Gunicorn (gunicorn app:app) y no python app.py.Verifica que no estés subiendo imágenes de 20MB. El sistema redimensiona internamente, pero la subida consume RAM.Desarrollado para Electro Sur Este S.A.A.
