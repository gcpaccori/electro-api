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
DEVICE = torch.device('cpu') 
# Parámetros solicitados por el usuario
CONF_DISP = 0.25
IOU_DISP = 0.5
CONF_DIG = 0.3    # Tu configuración
IOU_DIG = 0.3     # Tu configuración (ajustado)
WARP_W = 400
WARP_H = 150

# --- CARGA DE MODELOS ---
try:
    print(f"🔄 Electro API: Cargando modelos en {DEVICE}...")
    display_model = YOLO('display_detection.pt')
    digit_model = YOLO('digit_recognition.pt')
    print("✅ Modelos listos.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
    display_model = None
    digit_model = None

def numpy_to_base64(img_array):
    success, buffer = cv2.imencode('.jpg', img_array)
    if not success: return ""
    return base64.b64encode(buffer).decode('utf-8')

def get_reading_value(boxes, names):
    if not boxes: return "Error"
    # Ordenar dígitos de izquierda a derecha (coordenada X)
    digits = []
    for box in boxes:
        x1 = float(box.xyxy[0][0])
        cls = int(box.cls[0])
        label = names[cls]
        digits.append({"x": x1, "label": label})
    
    digits.sort(key=lambda k: k['x'])
    
    # Concatenar
    val = ""
    for d in digits:
        lbl = str(d['label'])
        # Ajuste para el punto decimal si tu clase se llama '10' o 'dot'
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
        return jsonify({"success": False, "error": "Modelos no cargados"}), 500
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "Falta imagen"}), 400

    try:
        file = request.files['image']
        pil_img = Image.open(io.BytesIO(file.read())).convert("RGB")
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        response_data = {
            "success": False,
            "reading": "No detectado",
            "debug_original": "",
            "debug_warp": ""
        }

        # 1. Detectar Display
        # Aplicamos tus parámetros de display
        disp_res = display_model(orig_img, conf=CONF_DISP, iou=IOU_DISP, device='cpu', verbose=False)
        
        if len(disp_res[0].boxes) > 0:
            # Tomar el mejor display
            box = disp_res[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Dibujar en original
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # 2. Corrección de Perspectiva (WARP)
            crop = orig_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            if crop.size > 0:
                # Forzar tamaño fijo 400x150
                warp_img = cv2.resize(crop, (WARP_W, WARP_H))
                
                # 3. Detectar Dígitos (con tus parámetros AJUSTADOS)
                # iou=0.3 es clave aquí para evitar dobles cajas
                dig_res = digit_model(warp_img, conf=CONF_DIG, iou=IOU_DIG, device='cpu', verbose=False)
                
                # Dibujar en Warp
                for dbox in dig_res[0].boxes:
                    dx1, dy1, dx2, dy2 = map(int, dbox.xyxy[0])
                    lbl = digit_model.names[int(dbox.cls[0])]
                    cv2.rectangle(warp_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
                    cv2.putText(warp_img, str(lbl), (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                # Obtener valor
                val = get_reading_value(dig_res[0].boxes, digit_model.names)
                
                response_data["success"] = True
                response_data["reading"] = val
                response_data["debug_warp"] = numpy_to_base64(warp_img)

        response_data["debug_original"] = numpy_to_base64(orig_img)
        return jsonify(response_data)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
