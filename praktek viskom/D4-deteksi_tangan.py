import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
# Konfigurasi: 1 tangan, kepercayaan deteksi 0.7, kepercayaan pelacakan 0.5
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.7, 
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ID untuk ujung jari (Tips)
# [Ibu Jari, Telunjuk, Tengah, Manis, Kelingking]
tip_ids = [4, 8, 12, 16, 20]

# ID untuk sendi "pip" (proximal interphalangeal) atau sendi di bawah ujung
# Ini akan jadi pembanding
pip_ids = [3, 6, 10, 14, 18]

# Buka webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

print("Membuka kamera... Tekan 'Esc' untuk keluar.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Tidak bisa membaca frame dari kamera.")
        break

    # 1. Balik gambar (flip) secara horizontal untuk tampilan "cermin"
    image = cv2.flip(image, 1)

    # 2. Konversi BGR ke RGB (karena MediaPipe perlu RGB)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Set gambar agar tidak bisa ditulis (read-only) untuk performa
    image_rgb.flags.writeable = False

    # 3. Proses gambar dan deteksi tangan
    results = hands.process(image_rgb)

    # Set gambar agar bisa ditulis kembali
    image_rgb.flags.writeable = True
    # Konversi balik ke BGR untuk ditampilkan OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR) # Gunakan ini untuk menggambar

    finger_count = 0
    
    # 4. Logika Penghitungan Jari
    if results.multi_hand_landmarks:
        # Kita atur max_num_hands=1, jadi kita ambil yg pertama [0]
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Dapatkan list semua landmark
        lm_list = hand_landmarks.landmark
        
        # Dapatkan label Tangan (Kiri atau Kanan) dari perspektif cermin
        # Ini penting untuk logika ibu jari
        hand_label = results.multi_handedness[0].classification[0].label

        fingers = []

        # --- Logika Ibu Jari ---
        # Logika khusus untuk ibu jari (horizontal)
        # Kita cek berdasarkan label "Kanan" atau "Kiri" (di tampilan cermin)
        if hand_label == "Right":
            # Tangan Kanan (visual): Ujung (4) harus di KIRI (x lebih kecil) dari sendi (3)
            if lm_list[tip_ids[0]].x < lm_list[pip_ids[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        elif hand_label == "Left":
            # Tangan Kiri (visual): Ujung (4) harus di KANAN (x lebih besar) dari sendi (3)
            if lm_list[tip_ids[0]].x > lm_list[pip_ids[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)

        # --- Logika 4 Jari Lainnya ---
        # (Telunjuk, Tengah, Manis, Kelingking)
        for i in range(1, 5): # Mulai dari 1 (Telunjuk) sampai 4 (Kelingking)
            # Cek jika ujung jari (tip) Y lebih KECIL (lebih atas) dari sendi (pip) Y
            if lm_list[tip_ids[i]].y < lm_list[pip_ids[i]].y:
                fingers.append(1)
            else:
                fingers.append(0)

        # Jumlahkan semua jari yang terdeteksi "up" (nilai 1)
        finger_count = sum(fingers)

        # 5. Gambar landmark tangan
        mp_drawing.draw_landmarks(
            image_bgr,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

    # 6. Tampilkan hasil hitungan
    cv2.putText(image_bgr, f'Jari: {finger_count}', 
                (20, 70), cv2.FONT_HERSHEY_PLAIN, 
                3, (0, 0, 255), 3)

    # 7. Tampilkan gambar ke layar
    cv2.imshow('Penghitung Jari - MediaPipe', image_bgr)

    # Keluar jika tombol 'Esc' (27) ditekan
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Bersihkan
print("Menutup program...")
cap.release()
hands.close()
cv2.destroyAllWindows()