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

# Ajustes de YOLO (Base)
CONF_DIG = 0.30  
IOU_DIG = 0.20   

# Dimensiones estándar
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

def solve_overlapping_digits(digits_data, min_dist=20):
    """
    Recibe una lista de diccionarios de DÍGITOS (sin puntos).
    Si dos dígitos tienen sus centros horizontales a menos de 'min_dist' píxeles,
    se elimina el de menor confianza.
    min_dist=20 es bueno para una imagen de 400px de ancho.
    """
    if not digits_data:
        return []

    # 1. Ordenar por CONFIANZA (el más seguro primero)
    # Así nos aseguramos de que si compiten 0 (90%) y 7 (40%), gane el 0.
    digits_data.sort(key=lambda x: x['conf'], reverse=True)

    final_digits = []
    
    for current_digit in digits_data:
        is_duplicate = False
        for accepted_digit in final_digits:
            # Calcular distancia entre centros X
            dist = abs(current_digit['x'] - accepted_digit['x'])
            
            # Si están demasiado pegados, es una colisión
            if dist < min_dist:
                is_duplicate = True
                break # Descartar 'current_digit' (porque tiene menos confianza que 'accepted')
        
        if not is_duplicate:
            final_digits.append(current_digit)

    return final_digits

def filter_dots_logic(sorted_items):
    """
    Lógica para borrar puntos duplicados o erróneos, conservando el último.
    """
    dots_indices = [i for i, d in enumerate(sorted_items) if d['is_dot']]
    
    if len(dots_indices) > 1:
        last_dot_idx = dots_indices[-1]
        filtered = []
        for i, d in enumerate(sorted_items):
            if i in dots_indices and i != last_dot_idx:
                continue 
            filtered.append(d)
        return filtered
    
    return sorted_items

def get_reading_from_crop(img_crop):
    """
    Ejecuta detección y aplica LIMPIEZA MANUAL de superposiciones.
    """
    results = digit_model(img_crop, conf=CONF_DIG, iou=IOU_DIG, device='cpu', verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        return "", 0.0, 0, img_crop

    raw_digits = [] # Solo números (0-9)
    raw_dots = []   # Solo puntos (dot, 10, point)
    names = digit_model.names

    # 1. Extracción y Separación
    for box in boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = str(names[cls])
        
        center_x = (x1 + x2) / 2
        
        item = {
            "x": center_x, 
            "label": label, 
            "conf": conf, 
            "box": box,
            "is_dot": label in ['10', 'dot', 'point', '.'] # Flag para identificar puntos
        }

        if item['is_dot']:
            raw_dots.append(item)
        else:
            raw_digits.append(item)

    # 2. LIMPIEZA DE DÍGITOS (Aquí eliminamos el 7 fantasma superpuesto al 0)
    # Usamos un radio de 20px (ajustable) para considerar conflicto
    clean_digits = solve_overlapping_digits(raw_digits, min_dist=20)

    # 3. FUSIÓN: Juntamos los dígitos limpios con TODOS los puntos originales
    # (Así garantizamos no afectar al dot)
    all_items = clean_digits + raw_dots

    # 4. Ordenar todo de Izquierda a Derecha
    all_items.sort(key=lambda k: k['x'])

    # 5. Filtrar puntos sobrantes (regla del último punto)
    final_items = filter_dots_logic(all_items)

    # 6. Construir String y Dibujar
    val_str = ""
    total_conf = 0.0
    annotated_img = img_crop.copy()

    for d in final_items:
        lbl = d['label']
        if d['is_dot']:
            val_str += "."
        else:
            val_str += lbl
        
        total_conf += d['conf']

        # Dibujar (Visual)
        box = d['box']
        dx1, dy1, dx2, dy2 = map(int, box.xyxy[0])
        
        # Color diferente para puntos y números
        color = (0, 0, 255) if d['is_dot'] else (255, 0, 0) # Rojo para números, Azul(ish) para puntos en BGR
        
        cv2.rectangle(annotated_img, (dx1, dy1), (dx2, dy2), color, 2)
        cv2.putText(annotated_img, lbl, (dx1, dy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    avg_conf = total_conf / len(final_items) if len(final_items) > 0 else 0
    return val_str, avg_conf, len(final_items), annotated_img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
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

        # Detección Display
        disp_res = display_model(orig_img, conf=CONF_DISP, iou=IOU_DISP, device='cpu', verbose=False)
        
        if len(disp_res[0].boxes) > 0:
            response["display_detected"] = True
            
            box = disp_res[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if include_visuals:
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

            crop = orig_img[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            
            if crop.size > 0:
                h, w = crop.shape[:2]
                ratio = w / h
                
                final_reading = ""
                final_warp_img = crop
                
                # --- GEOMETRÍA INTELIGENTE ---
                if ratio < 0.85: # Vertical
                    rot_left = cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE), (STD_W, STD_H))
                    read_L, conf_L, count_L, img_L = get_reading_from_crop(rot_left)

                    rot_right = cv2.resize(cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE), (STD_W, STD_H))
                    read_R, conf_R, count_R, img_R = get_reading_from_crop(rot_right)

                    # Score ponderado
                    if (count_L * 10 + conf_L) >= (count_R * 10 + conf_R):
                        final_reading, final_warp_img = read_L, img_L
                    else:
                        final_reading, final_warp_img = read_R, img_R

                elif 0.85 <= ratio <= 1.3: # Cuadrado
                    target_w = 350
                    scale = target_w / w
                    target_h = int(h * scale)
                    resized_square = cv2.resize(crop, (target_w, target_h))
                    final_reading, _, _, img_sq = get_reading_from_crop(resized_square)
                    final_warp_img = img_sq

                else: # Horizontal
                    warp_std = cv2.resize(crop, (STD_W, STD_H))
                    final_reading, _, _, img_std = get_reading_from_crop(warp_std)
                    final_warp_img = img_std

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
