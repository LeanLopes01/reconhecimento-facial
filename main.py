import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_V4L'] = '1'
os.environ['DEEPFACE_BACKEND'] = 'tf'

from deepface import DeepFace
import cv2
import time
import logging
import threading
from queue import Queue
import numpy as np
from datetime import datetime

# Configurações do MySQL
MYSQL_CONFIG = {
    'host': 'localhost',  # ou o endereço do seu servidor MySQL
    'user': 'face',  # seu usuário do MySQL
    'password': os.getenv('MYSQL_PASSWORD', 'your-mysql-password'),  # sua senha do MySQL
    'database': 'face_database'  # nome do banco de dados
}

# Configurações
ANALYSIS_INTERVAL = 3
MAX_FACES = 4
DB_PATH = "database"
RECOGNITION_THRESHOLD = 0.6
LOG_FILE = "facial_recognition.log"
WINDOW_NAME = "Reconhecimento Facial"

# Configurar logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# Inicializar banco de dados MySQL
def init_database():
    try:
        # Importar a biblioteca MySQL
        import mysql.connector
        
        # Conectar ao banco de dados
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        logger.info("Conectado ao banco de dados MySQL com sucesso")
        return conn
    except Exception as e:
        logger.error(f"Erro ao conectar ao MySQL: {str(e)}")
        exit()

db_conn = init_database()

# Inicializa webcam
def init_camera():
    for i in range(0, 8):
        cap = cv2.VideoCapture(f'/dev/video{i}')
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
            logger.info(f"Webcam encontrada em /dev/video{i}")
            return cap
    logger.error("Nenhuma webcam detectada. Saindo...")
    exit()

cap = init_camera()

# Filas para processamento
face_queue = Queue(maxsize=1)
db_queue = Queue()
results = []
results_lock = threading.Lock()
last_saved = {}

def save_to_database(name, confidence):
    try:
        current_time = time.time()
        if name in last_saved and current_time - last_saved[name] < 30:
            return

        cursor = db_conn.cursor()
        # Corrigir: usar apenas dois valores (nome e confiança)
        cursor.execute("INSERT INTO face_reconhecidas (nome, confianca) VALUES (%s, %s)",
                     (name, confidence))  # Remover datetime.now() daqui
        db_conn.commit()
        last_saved[name] = current_time
        logger.info(f"Registro salvo no BD: {name} ({confidence*100:.1f}%)")
    except Exception as e:
        logger.error(f"Erro ao salvar no BD: {str(e)}")

def database_worker():
    while True:
        try:
            data = db_queue.get()
            if data is None:
                break
            name, confidence = data
            save_to_database(name, confidence)
        except Exception as e:
            logger.error(f"Erro no worker do BD: {str(e)}")

def get_identity(face_image):
    try:
        dfs = DeepFace.find(
            img_path=face_image,
            db_path=DB_PATH,
            model_name="VGG-Face",
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
            distance_metric='cosine'
        )

        if dfs and not dfs[0].empty:
            best_match = dfs[0].iloc[0]
            distance_columns = [col for col in best_match.index if 'distance' in col.lower() or 'cosine' in col.lower()]
            
            if distance_columns:
                distance_col = distance_columns[0]
                distance = best_match[distance_col]
                
                if distance < RECOGNITION_THRESHOLD:
                    identity = os.path.basename(os.path.dirname(best_match['identity']))
                    confidence = 1 - distance
                    return identity, confidence
    except Exception as e:
        logger.error(f"Erro no reconhecimento: {str(e)}")
    
    return "Desconhecido", 0

def process_faces():
    while True:
        try:
            frame, face_objs = face_queue.get()
            if frame is None:
                break
                
            current_results = []
            
            for face_obj in face_objs[:MAX_FACES]:
                if 'confidence' in face_obj and face_obj['confidence'] > 0.85:
                    try:
                        region = face_obj['facial_area']
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if face_roi.size == 0:
                            continue
                            
                        identity, confidence = get_identity(face_roi)
                        
                        if identity != "Desconhecido" and confidence > 0.6:
                            logger.info(f"Reconhecido: {identity} ({confidence*100:.1f}%)")
                            db_queue.put((identity, confidence))
                        
                        current_results.append({
                            'identity': identity,
                            'confidence': confidence,
                            'region': region
                        })
                    except Exception as e:
                        logger.error(f"Erro no processamento do rosto: {e}")
            
            with results_lock:
                global results
                results = current_results
                
        except Exception as e:
            logger.error(f"Erro na thread de processamento: {e}")

# Iniciar threads
processing_thread = threading.Thread(target=process_faces, daemon=True)
db_thread = threading.Thread(target=database_worker, daemon=True)
processing_thread.start()
db_thread.start()

# Janela de exibição
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 800, 600)

last_analysis = time.time() - ANALYSIS_INTERVAL

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Erro ao capturar frame")
            time.sleep(0.1)
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        processing_frame = frame.copy()
        display_frame = frame.copy()

        with results_lock:
            current_results = results.copy()

        for result in current_results:
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            identity = result['identity']
            confidence = result['confidence']

            color = (0, 255, 0) if identity != "Desconhecido" else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            label = f"{identity} ({confidence*100:.1f}%)"
            cv2.putText(display_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow(WINDOW_NAME, display_frame)

        current_time = time.time()
        if current_time - last_analysis > ANALYSIS_INTERVAL:
            try:
                last_analysis = current_time
                face_objs = DeepFace.extract_faces(
                    img_path=processing_frame,
                    detector_backend='opencv',
                    enforce_detection=False,
                    align=False
                )
                
                if not face_queue.full():
                    face_queue.put((processing_frame, face_objs))
                    
            except Exception as e:
                logger.error(f"Erro na detecção de rostos: {str(e)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    logger.info("\nPrograma interrompido")
except Exception as e:
    logger.exception("Erro fatal")
finally:
    face_queue.put((None, None))
    db_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    db_conn.close()
    logger.info("Recursos liberados")