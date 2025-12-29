import os
import io
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image

# Necesitamos definir la carpeta de templates para el frontend
app = Flask(__name__, template_folder='templates')

# --- CARGA DE MODELOS ---
try:
    print("Cargando modelos...")
    display_model = YOLO('display_detection.pt')
    digit_model = YOLO('digit_recognition.pt')
    print("Modelos listos.")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    display_model = None
    digit_model = None

def numpy_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# --- RUTA NUEVA PARA LA INTERFAZ GRÁFICA ---
@app.route('/', methods=['GET'])
def index():
    # Esto carga el archivo index.html de la carpeta templates
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not display_model or not digit_model:
        return jsonify({"success": False, "error": "Modelos no cargados"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files['image']
    image_bytes = file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    final_detections = []
    display_found = False
    offset_x, offset_y = 0, 0
    working_img = orig_img.copy()

    # 1. Detectar Display
    disp_results = display_model(orig_img, conf=0.4, verbose=False)
    if len(disp_results[0].boxes) > 0:
        display_found = True
        box = disp_results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Clamping
        h, w = orig_img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        offset_x, offset_y = x1, y1
        working_img = orig_img[y1:y2, x1:x2]
        # Dibujar cuadro verde (Display)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 2. Detectar Dígitos
    digit_results = digit_model(working_img, conf=0.25, verbose=False)
    for result in digit_results:
        for box in result.boxes:
            dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = digit_model.names[int(box.cls[0])]

            # Re-map coordenadas
            fx1 = dx1 + offset_x
            fy1 = dy1 + offset_y
            fx2 = dx2 + offset_x
            fy2 = dy2 + offset_y

            final_detections.append({
                "label": label,
                "confidence": f"{conf:.2f}",
                "box": [fx1, fy1, fx2, fy2]
            })

            # Dibujar cuadro rojo (Dígitos)
            cv2.rectangle(orig_img, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            cv2.putText(orig_img, str(label), (fx1, fy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return jsonify({
        "success": True,
        "detections": final_detections,
        "processed_image": numpy_to_base64(orig_img)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)