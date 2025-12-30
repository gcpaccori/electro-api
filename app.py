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

# AJUSTES PARA DÍGITOS (CRÍTICO PARA EVITAR DOBLES DETECCIONES)
CONF_DIG = 0.30  # Confianza mínima
IOU_DIG = 0.20   # Bajado a 0.2 para eliminar cajas superpuestas (ej: 0 y 7 encimados)

# Dimensiones estándar solo para displays horizontales
STD_W = 400
STD_H = 150

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

def filter_dots(digits_list):
    """
    Recibe la lista de dígitos detectados y ordenados.
    Si hay múltiples puntos, conserva solo el último (más a la derecha).
    """
    dots_indices = [i for i, d in enumerate(digits_list) if d['label'] in ['10', 'dot', 'point']]
    
    if len(dots_indices) > 1:
        last_dot_idx = dots_indices[-1]
        filtered = []
        for i, d in enumerate(digits_list):
            # Si es un punto y NO es el último, lo saltamos
            if i in dots_indices and i != last_dot_idx:
                continue 
            filtered.append(d)
        return filtered
    
    return digits_list

def get_reading_from_crop(img_crop):
    """
    Función auxiliar: Ejecuta detección, filtra overlap y ordena por CENTRO X.
    """
    # Ejecutar modelo de dígitos con IOU estricto (0.2)
    results = digit_model(img_crop, conf=CONF_DIG, iou=IOU_DIG, device='cpu', verbose=False)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return "", 0.0, 0, img_crop

    digits_found = []
    total_conf = 0.0
    names = digit_model.names

    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = names[cls]
        
        # --- CRÍTICO: USAR CENTRO X PARA ORDENAR ---
        center_x = (x1 + x2) / 2
        
        digits_found.append({"x": center_x, "label": label, "conf": conf, "box": box})
        total_conf += conf

    # Ordenar de izquierda a derecha usando el CENTRO
    digits_found.sort(key=lambda k: k['x'])

    # Filtrar puntos sobrantes
    digits_found = filter_dots(digits_found)

    # Construir string
    val_str = ""
    for d in digits_found:
        lbl = str(d['label'])
        if lbl in ['10', 'dot', 'point']:
            val_str += "."
        else:
            val_str += lbl

    # Dibujar resultados sobre la imagen (Debugging visual)
    annotated_img = img_crop.copy()
    for d in digits_found:
        box = d['box']
        dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated_img, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
        cv2.putText(annotated_img, str(d['label']), (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    avg_conf = total_conf / len(digits_found) if len(digits_found) > 0 else 0
    return val_str, avg_conf, len(digits_found), annotated_img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Inicializar variable para manejo de errores seguro
    filename = "unknown"

    if not display_model or not digit_model:
        return jsonify({"filename": "error", "display_detected": False, "reading": "Modelos no cargados"}), 500

    if 'image' not in request.files:
        return jsonify({"filename": "error", "display_detected": False, "reading": "Falta imagen"}), 400

    include_visuals = request.form.get('include_visuals') == 'true'

    try:
        file = request.files['image']
        filename = secure_filename(file.filename)
        file_bytes = file.read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        response = {
            "filename": filename,
            "display_detected": False,
            "reading": "desactivado"
        }

        # 1. Detectar Display
        disp_res = display_model(orig_img, conf=CONF_DISP, iou=IOU_DISP, device='cpu', verbose=False)
        
        if len(disp_res[0].boxes) > 0:
            response["display_detected"] = True
            
            # Tomar el mejor display
            box = disp_res[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Visualización (Solo Web)
            if include_visuals:
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Recortar Display
            crop = orig_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            
            if crop.size > 0:
                h, w = crop.shape[:2]
                ratio = w / h
                
                final_reading = ""
                final_warp_img = crop
                
                # --- LÓGICA DE GEOMETRÍA INTELIGENTE ---

                # CASO 1: VERTICAL (Alto > Ancho)
                if ratio < 0.85: 
                    # Probar rotación Izquierda (-90)
                    rot_left = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    rot_left = cv2.resize(rot_left, (STD_W, STD_H))
                    read_L, conf_L, count_L, img_L = get_reading_from_crop(rot_left)

                    # Probar rotación Derecha (+90)
                    rot_right = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                    rot_right = cv2.resize(rot_right, (STD_W, STD_H))
                    read_R, conf_R, count_R, img_R = get_reading_from_crop(rot_right)

                    # Comparar ganadores
                    score_L = count_L * 10 + conf_L
                    score_R = count_R * 10 + conf_R

                    if score_L >= score_R:
                        final_reading = read_L
                        final_warp_img = img_L
                    else:
                        final_reading = read_R
                        final_warp_img = img_R

                # CASO 2: CUADRADO (Ratio cercano a 1)
                elif 0.85 <= ratio <= 1.3:
                    # Escalar proporcionalmente (NO estirar)
                    target_w = 350
                    scale = target_w / w
                    target_h = int(h * scale)
                    resized_square = cv2.resize(crop, (target_w, target_h))
                    
                    final_reading, _, _, img_sq = get_reading_from_crop(resized_square)
                    final_warp_img = img_sq

                # CASO 3: HORIZONTAL (Normal)
                else:
                    # Estirar a formato estándar 400x150
                    warp_std = cv2.resize(crop, (STD_W, STD_H))
                    final_reading, _, _, img_std = get_reading_from_crop(warp_std)
                    final_warp_img = img_std

                # Asignar resultados
                response["reading"] = final_reading if final_reading else "ilegible"
                
                if include_visuals:
                    response["debug_warp"] = numpy_to_base64(final_warp_img)
        
        if include_visuals:
            response["debug_original"] = numpy_to_base64(orig_img)

        return jsonify(response)

    except Exception as e:
        print(f"Error procesando {filename}: {e}")
        return jsonify({"filename": filename, "display_detected": False, "reading": "error_servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
