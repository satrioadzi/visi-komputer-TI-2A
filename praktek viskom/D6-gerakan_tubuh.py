import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Fungsi Helper ---

def calculate_angle(a, b, c):
    """Menghitung sudut (di B) dari 3 titik A, B, C."""
    # Konversi ke array numpy
    p1 = np.array([a.x, a.y]) # Titik A (misal: Pinggul)
    p2 = np.array([b.x, b.y]) # Titik B (Vertex, misal: Lutut)
    p3 = np.array([c.x, c.y]) # Titik C (misal: Pergelangan Kaki)
    
    # Hitung vektor
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Hitung dot product dan magnitudo
    dot_prod = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    
    # Hindari pembagian dengan nol
    if mag_v1 == 0 or mag_v2 == 0:
        return 0
    
    # Hitung cos theta dan konversi ke derajat
    cos_theta = dot_prod / (mag_v1 * mag_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # Jaga nilai antara -1 dan 1
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def get_distance(p1, p2):
    """Menghitung jarak Euclidean antara 2 titik landmark."""
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# --- Inisialisasi Variabel ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Variabel state machine
counter = 0
status = "UP"  # Status awal
mode = "SQUAT" # Mode awal

# Threshold (sesuai permintaan)
SQUAT_DOWN_THRESHOLD = 80
SQUAT_UP_THRESHOLD = 160

# Threshold untuk rasio Push-up (ini perlu disesuaikan/tuning)
PUSHUP_DOWN_THRESHOLD = 0.4
PUSHUP_UP_THRESHOLD = 0.7

print(f"Mode Awal: {mode}. Tekan 'm' untuk ganti mode, 'Esc' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Balik gambar (flip) dan konversi warna
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Proses deteksi pose
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    current_value = 0.0 # Nilai (sudut atau rasio) saat ini

    # 3. Ekstraksi Landmark dan Logika
    try:
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- LOGIKA SQUAT (Permintaan #2) ---
            if mode == "SQUAT":
                # Ambil landmark lutut
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                # Hitung sudut
                angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
                
                # Gunakan rata-rata untuk stabilitas
                current_value = (angle_left_knee + angle_right_knee) / 2
                
                # State Machine (Permintaan #4)
                if current_value < SQUAT_DOWN_THRESHOLD:
                    status = "DOWN"
                if current_value > SQUAT_UP_THRESHOLD and status == "DOWN":
                    status = "UP"
                    counter += 1
                    print(f"SQUAT Rep: {counter}")

            # --- LOGIKA PUSH-UP (Permintaan #3) ---
            elif mode == "PUSHUP":
                # Ambil landmark bahu, pergelangan, pinggul
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                # Hitung jarak
                dist_sw_left = get_distance(left_shoulder, left_wrist)
                dist_sh_left = get_distance(left_shoulder, left_hip)
                
                dist_sw_right = get_distance(right_shoulder, right_wrist)
                dist_sh_right = get_distance(right_shoulder, right_hip)

                # Hitung rasio (pastikan tidak ada pembagian nol)
                ratio_left = dist_sw_left / dist_sh_left if dist_sh_left > 0 else 0
                ratio_right = dist_sw_right / dist_sh_right if dist_sh_right > 0 else 0
                
                current_value = (ratio_left + ratio_right) / 2
                
                # State Machine (Permintaan #4)
                if current_value < PUSHUP_DOWN_THRESHOLD:
                    status = "DOWN"
                if current_value > PUSHUP_UP_THRESHOLD and status == "DOWN":
                    status = "UP"
                    counter += 1
                    print(f"PUSHUP Rep: {counter}")

            # 4. Gambar landmark
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

    except Exception as e:
        print(f"Error memproses landmark: {e}")
        pass # Lanjut ke frame berikutnya

    # 5. Tampilkan Status Box (Permintaan #5)
    # Latar belakang box
    cv2.rectangle(image_bgr, (0, 0), (500, 100), (20, 20, 20), -1)

    # Teks Mode
    cv2.putText(image_bgr, f"MODE: {mode} (Tekan 'm')",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Teks Repetisi
    cv2.putText(image_bgr, "REPS",
                (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, str(counter),
                (70, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Teks Status
    cv2.putText(image_bgr, "STATUS",
                (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(image_bgr, status,
                (280, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0) if status == "UP" else (0, 0, 255), 2, cv2.LINE_AA)

    # Tampilkan gambar
    cv2.imshow('Penghitung Aktivitas Fisik', image_bgr)

    # Kontrol Keyboard
    key = cv2.waitKey(5) & 0xFF
    if key == 27: # 'Esc'
        break
    elif key == ord('m'):
        if mode == "SQUAT":
            mode = "PUSHUP"
        else:
            mode = "SQUAT"
        counter = 0 # Reset hitungan
        status = "UP" # Reset status
        print(f"Mode diubah ke: {mode}")

# Bersihkan
print("Menutup program...")
cap.release()
pose.close()
cv2.destroyAllWindows()