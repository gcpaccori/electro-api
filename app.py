import io
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
from PIL import Image
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')

# --- CONFIGURACIÓN ---
DEVICE = torch.device('cpu') 
CONF_DISP = 0.25
IOU_DISP = 0.5
CONF_DIG = 0.3
IOU_DIG = 0.3
WARP_W = 400
WARP_H = 150

# --- CARGA DE MODELOS ---
try:
    print(f"🔄 Electro API: Cargando modelos en {DEVICE}...")
    display_model = YOLO('display_detection.pt')
    digit_model = YOLO('digit_recognition.pt')
    print("✅ Modelos listos.")
except Exception as e:
    print(f"❌ Error crítico: {e}")
    display_model = None
    digit_model = None

def numpy_to_base64(img_array):
    success, buffer = cv2.imencode('.jpg', img_array)
    if not success: return ""
    return base64.b64encode(buffer).decode('utf-8')

def get_reading_value(boxes, names):
    if not boxes: return ""
    digits = []
    for box in boxes:
        x1 = float(box.xyxy[0][0])
        cls = int(box.cls[0])
        label = names[cls]
        digits.append({"x": x1, "label": label})
    
    digits.sort(key=lambda k: k['x'])
    
    val = ""
    for d in digits:
        lbl = str(d['label'])
        if lbl in ['10', 'dot', 'point']: 
            val += "."
        else:
            val += lbl
    return val

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not display_model or not digit_model:
        return jsonify({"filename": "error", "display_detected": False, "reading": "Modelos no cargados"}), 500

    if 'image' not in request.files:
        return jsonify({"filename": "error", "display_detected": False, "reading": "Falta imagen"}), 400

    # Flag para saber si devolvemos imágenes (Web) o solo JSON (API masiva)
    include_visuals = request.form.get('include_visuals') == 'true'

    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        
        # Leer imagen
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Respuesta Base
        response = {
            "filename": filename,
            "display_detected": False,
            "reading": "desactivado" # Por defecto
        }

        # 1. Detectar Display
        disp_res = display_model(orig_img, conf=CONF_DISP, iou=IOU_DISP, device='cpu', verbose=False)
        
        if len(disp_res[0].boxes) > 0:
            response["display_detected"] = True
            
            # Procesar Display
            box = disp_res[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # (Solo si es Web) Dibujar en original
            if include_visuals:
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Recorte y Warp
            crop = orig_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            
            if crop.size > 0:
                warp_img = cv2.resize(crop, (WARP_W, WARP_H))
                
                # 2. Detectar Dígitos
                dig_res = digit_model(warp_img, conf=CONF_DIG, iou=IOU_DIG, device='cpu', verbose=False)
                
                # Obtener Valor
                val = get_reading_value(dig_res[0].boxes, digit_model.names)
                response["reading"] = val if val else "ilegible"

                # (Solo si es Web) Generar imágenes Base64
                if include_visuals:
                    for dbox in dig_res[0].boxes:
                        dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0])
                        lbl = digit_model.names[int(dbox.cls[0])]
                        cv2.rectangle(warp_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                        cv2.putText(warp_img, str(lbl), (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    
                    response["debug_warp"] = numpy_to_base64(warp_img)
        
        # (Solo si es Web) Adjuntar original procesada
        if include_visuals:
            response["debug_original"] = numpy_to_base64(orig_img)

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"filename": filename, "display_detected": False, "reading": "error_servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
