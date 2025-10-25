import cv2
import mediapipe as mp
import math
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print("Membuka kamera... Tekan 'Esc' untuk keluar.")

# --- Fungsi Helper untuk Geometri ---
def get_distance(p1, p2):
    """Menghitung jarak Euclidean antara dua landmark"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

OK_DIST_THRESHOLD = 0.08      # Jarak antara ujung ibu jari dan telunjuk
ROCK_AVG_DIST_THRESHOLD = 0.25  # Jarak rata-rata ujung jari ke pergelangan (Batu)
PAPER_AVG_DIST_THRESHOLD = 0.45 # Jarak rata-rata ujung jari ke pergelangan (Kertas)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start_time = time.time()
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    gesture = "Tidak Dikenali"

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = hand_landmarks.landmark

        # --- 1. Ambil Landmark Kunci (sesuai permintaan) ---
        wrist_lm = lm_list[0]
        thumb_tip_lm = lm_list[4]
        index_tip_lm = lm_list[8]
        middle_tip_lm = lm_list[12]
        ring_tip_lm = lm_list[16]
        pinky_tip_lm = lm_list[20]

        # --- 2. Hitung Status Jari (Terangkat/Tertutup) ---
        # (Menggunakan logika Y-koordinat relatif)
        is_thumb_up = thumb_tip_lm.y < lm_list[3].y
        is_index_up = index_tip_lm.y < lm_list[6].y
        is_middle_up = middle_tip_lm.y < lm_list[10].y
        is_ring_up = ring_tip_lm.y < lm_list[14].y
        is_pinky_up = pinky_tip_lm.y < lm_list[18].y

        # --- 3. Hitung Jarak Rata-rata (untuk Rock/Paper) ---
        dist_index = get_distance(index_tip_lm, wrist_lm)
        dist_middle = get_distance(middle_tip_lm, wrist_lm)
        dist_ring = get_distance(ring_tip_lm, wrist_lm)
        dist_pinky = get_distance(pinky_tip_lm, wrist_lm)
        
        # Hitung rata-rata jarak dari 4 ujung jari ke pergelangan tangan
        avg_dist = (dist_index + dist_middle + dist_ring + dist_pinky) / 4

        dist_ok = get_distance(thumb_tip_lm, index_tip_lm)
        if (dist_ok < OK_DIST_THRESHOLD and 
            is_middle_up and is_ring_up and is_pinky_up):
            gesture = "OK"

        elif (is_thumb_up and 
              not is_index_up and not is_middle_up and 
              not is_ring_up and not is_pinky_up):
            gesture = "Thumbs Up"

        elif (is_index_up and is_middle_up and 
              not is_ring_up and not is_pinky_up):
            gesture = "Gunting (Scissors)"

        elif (not is_index_up and not is_middle_up and 
              not is_ring_up and not is_pinky_up) or \
             (avg_dist < ROCK_AVG_DIST_THRESHOLD):
            gesture = "Batu (Rock)"

        elif (is_index_up and is_middle_up and 
              is_ring_up and is_pinky_up) or \
             (avg_dist > PAPER_AVG_DIST_THRESHOLD):
            gesture = "Kertas (Paper)"


        # Gambar landmark tangan
        mp_drawing.draw_landmarks(
            image_bgr,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

    # Hitung waktu pemrosesan
    end_time = time.time()
    processing_time_ms = (end_time - start_time) * 1000

    # Tampilkan hasil
    cv2.putText(image_bgr, f'Gestur: {gesture}', 
                (20, 70), cv2.FONT_HERSHEY_PLAIN, 
                3, (255, 0, 0), 3)
    
    cv2.putText(image_bgr, f'Time: {processing_time_ms:.2f} ms', 
                (20, 130), cv2.FONT_HERSHEY_PLAIN, 
                2, (0, 0, 255), 2)

    # Tampilkan gambar ke layar
    cv2.imshow('Pengenal Gestur Geometris', image_bgr)

    if cv2.waitKey(5) & 0xFF == 27:
        break

print("Menutup program...")
cap.release()
hands.close()
cv2.destroyAllWindows()