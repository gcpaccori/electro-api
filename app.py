import os
import io
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import torch

app = Flask(__name__, template_folder='templates')

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cpu') # Forzamos CPU para Render y tu PC
CONF_DISP = 0.25
CONF_DIG = 0.3
IOU_DIG = 0.3    # Tu ajuste solicitado
WARP_W = 400     # Ancho forzado del display
WARP_H = 150     # Alto forzado del display

# --- CARGA DE MODELOS ---
try:
    print(f"🔄 Cargando modelos en {DEVICE}...")
    # map_location es VITAL para evitar errores de GPU vs CPU
    display_model = YOLO('display_detection.pt')
    digit_model = YOLO('digit_recognition.pt')
    print("✅ Modelos listos.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    display_model = None
    digit_model = None

# --- UTILIDADES ---
def numpy_to_base64(img_array):
    success, buffer = cv2.imencode('.jpg', img_array)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')

def get_reading_value(boxes, names):
    """
    Toma las cajas detectadas, las ordena de izquierda a derecha (coordenada X)
    y construye el string del valor (ej: "128.5").
    """
    if not boxes:
        return "N/A"

    # Estructura: (x1, clase_id, nombre)
    digits_found = []
    for box in boxes:
        x1 = float(box.xyxy[0][0])
        cls_id = int(box.cls[0])
        label = names[cls_id]
        digits_found.append({"x": x1, "label": label})

    # Ordenar por posición X (izquierda a derecha)
    digits_found.sort(key=lambda k: k['x'])

    # Construir string
    result_str = ""
    for d in digits_found:
        if d['label'] == 'dot' or d['label'] == '10': # Ajusta según cómo se llame tu clase punto
            result_str += "."
        else:
            result_str += str(d['label'])
    
    return result_str

# --- RUTAS ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not display_model or not digit_model:
        return jsonify({"success": False, "error": "Modelos no cargados"}), 500

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No se subió imagen"}), 400

    try:
        # 1. Leer imagen
        file = request.files['image']
        image_bytes = file.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        final_data = {
            "success": False,
            "reading": "No detectado",
            "debug_original": "",
            "debug_warp": ""
        }

        # 2. Paso 1: Detectar Display
        # Usamos device='cpu' explícitamente
        disp_results = display_model(orig_img, conf=CONF_DISP, iou=0.5, device='cpu', verbose=False)
        
        if len(disp_results[0].boxes) > 0:
            # Tomar el display con mayor confianza
            best_box = disp_results[0].boxes[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            
            # Dibujar rectángulo en original
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 3. Paso 2: Corrección de Perspectiva (Crop & Resize)
            # Recortamos
            crop_img = orig_img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            
            # "Warp" -> Redimensionar a tamaño fijo para normalizar dígitos
            if crop_img.size > 0:
                warp_img = cv2.resize(crop_img, (WARP_W, WARP_H))
                
                # 4. Paso 3: Detectar Dígitos en la imagen Warp
                # Aquí aplicamos TUS parámetros: conf=0.3, iou=0.3
                digit_results = digit_model(warp_img, conf=CONF_DIG, iou=IOU_DIG, device='cpu', verbose=False)
                
                # Dibujar dígitos en la imagen Warp
                for box in digit_results[0].boxes:
                    dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = digit_model.names[cls]
                    
                    # Cuadro rojo
                    cv2.rectangle(warp_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(warp_img, f"{label}", (dx1, dy1-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # 5. Obtener el valor numérico
                read_value = get_reading_value(digit_results[0].boxes, digit_model.names)

                # Preparar respuesta
                final_data["success"] = True
                final_data["reading"] = read_value
                final_data["debug_warp"] = numpy_to_base64(warp_img)
        
        # Convertir original a base64
        final_data["debug_original"] = numpy_to_base64(orig_img)

        return jsonify(final_data)

    except Exception as e:
        print(f"ERROR PROCESANDO: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # TIMEOUT y GUNICORN se configuran en Render, no aquí.
    # Aquí solo local:
    app.run(host='0.0.0.0', port=5000, debug=True)
