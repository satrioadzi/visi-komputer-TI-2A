import cv2
import time
import numpy as np
import mediapipe as mp

# --- Landmark Indeks 6-Titik (Standar EAR) ---
# [p1_kiri, p2_atas, p3_atas, p4_kanan, p5_bawah, p6_bawah]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 387, 385, 263, 380, 373]

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Parameter ---
# Threshold EAR untuk rumus 6-titik (mungkin perlu disesuaikan)
EYE_AR_THRESHOLD = 0.25
# Jumlah frame berturut-turut untuk dianggap kedip
CLOSED_FRAMES_THRESHOLD = 2 # 2 atau 3 frame biasanya ideal

def calculate_ear(eye_indices, landmarks, w, h):
    """
    Menghitung Eye Aspect Ratio (EAR) menggunakan rumus 6-titik.
    Landmarks adalah daftar landmark yang dinormalisasi (0.0 - 1.0).
    w dan h adalah lebar dan tinggi frame.
    """
    # Ekstrak 6 titik (p1-p6) dalam koordinat piksel
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append(np.array([int(lm.x * w), int(lm.y * h)]))
    
    p1, p2, p3, p4, p5, p6 = points

    # Hitung jarak vertikal
    # ||p2 - p6||
    dist_v1 = np.linalg.norm(p2 - p6)
    # ||p3 - p5||
    dist_v2 = np.linalg.norm(p3 - p5)
    
    # Hitung jarak horizontal
    # ||p1 - p4||
    dist_h = np.linalg.norm(p1 - p4)

    # Hindari pembagian dengan nol
    if dist_h == 0:
        return 0.0

    # Rumus EAR
    ear = (dist_v1 + dist_v2) / (2.0 * dist_h)
    return ear

# --- Inisialisasi ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit('Tidak bisa membuka kamera')

# Spek visualisasi untuk kontur mata
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

with mp_face_mesh.FaceMesh(max_num_faces=1,
                           refine_landmarks=True, # Penting untuk 6 titik mata
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    
    blink_count = 0
    closed_frames_counter = 0 # Satu counter untuk kedua mata
    is_blinking = False       # State machine (False=Open, True=Closed)
    last_blink_time = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1) # Balik horizontal
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Proses frame
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            lm = face_landmarks.landmark

            # --- 1. Hitung EAR untuk setiap mata ---
            left_ear = calculate_ear(LEFT_EYE_INDICES, lm, w, h)
            right_ear = calculate_ear(RIGHT_EYE_INDICES, lm, w, h)

            # --- 2. Rata-rata EAR (Logika Utama yang Diperbaiki) ---
            avg_ear = (left_ear + right_ear) / 2.0

            # --- 3. Logika State Machine (Diperbaiki) ---
            if avg_ear < EYE_AR_THRESHOLD:
                # Mata tertutup
                closed_frames_counter += 1
                
                # Jika mata tertutup cukup lama, set state ke "blinking"
                if closed_frames_counter >= CLOSED_FRAMES_THRESHOLD:
                    is_blinking = True
            else:
                # Mata terbuka
                # Jika kita *baru saja* membuka mata setelah berkedip...
                if is_blinking:
                    blink_count += 1      # Tambah hitungan
                    last_blink_time = time.time()
                    is_blinking = False   # Reset state
                
                # Selalu reset counter jika mata terbuka
                closed_frames_counter = 0

            # --- Visualisasi ---
            # Gambar kontur mata (lebih baik dari 4 titik)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            # Tampilkan info EAR (untuk debugging)
            # cv2.putText(frame, f'EAR: {avg_ear:.2f}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Tampilkan info kedip (sudah benar dari kode Anda)
        cv2.putText(frame, f'Blink count: {blink_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if time.time() - last_blink_time < 0.5: # Tampilkan "Blink!" selama 0.5 detik
            cv2.putText(frame, 'Blink!', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Deteksi Kedip MediaPipe', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()