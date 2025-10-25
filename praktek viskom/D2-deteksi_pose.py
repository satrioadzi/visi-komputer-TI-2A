import cv2
import math
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("ERROR: Cannot open camera. Close other apps that may be using it or check camera permissions.")

try:
    detector = PoseDetector(staticMode=False,
                            modelComplexity=1,
                            smoothLandmarks=False,
                            enableSegmentation=False,
                            detectionCon=0.5,
                            trackCon=0.5)
except Exception as e:
    cap.release()
    raise SystemExit(f"ERROR: Failed to initialize PoseDetector: {e}")


def calculate_angle(a, b, c):
    # angle at point b formed by points a-b-c
    BAx = a[0] - b[0]
    BAy = a[1] - b[1]
    BCx = c[0] - b[0]
    BCy = c[1] - b[1]
    ang = math.degrees(math.atan2(BCy, BCx) - math.atan2(BAy, BAx))
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return int(ang)


try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            print("WARNING: Failed to read frame from camera. Exiting loop.")
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img, draw=True)

        # Jika ada landmark yang terdeteksi
        if lmList and bboxInfo:
            # Dapatkan pusat bounding box (jika tersedia)
            center = bboxInfo.get('center') if isinstance(bboxInfo, dict) else None
            if center:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Ambil koordinat landmark 11, 13, 15
            # Format lmList entries: [id, x, y]
            try:
                x11, y11 = int(lmList[11][1]), int(lmList[11][2])
                x13, y13 = int(lmList[13][1]), int(lmList[13][2])
                x15, y15 = int(lmList[15][1]), int(lmList[15][2])

                # Gambar titik dan garis antara 11 dan 15
                cv2.circle(img, (x11, y11), 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x15, y15), 5, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x11, y11), (x15, y15), (255, 0, 0), 2)

                # Hitung jarak Euclidean antara landmark 11 dan 15
                dist = int(math.hypot(x15 - x11, y15 - y11))
                cv2.putText(img, f"Dist:{dist}", (x13 + 10, y13 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Hitung sudut antara 11-13-15 (sudut di landmark 13)
                angle = calculate_angle((x11, y11), (x13, y13), (x15, y15))
                cv2.putText(img, f"Angle:{angle}", (x13 - 50, y13 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Periksa apakah sudut mendekati 50 derajat dengan offset 10
                isCloseAngle50 = abs(angle - 50) <= 10
                print(isCloseAngle50)
            except Exception:
                # Jika indeks landmark tidak tersedia, lewati
                pass

        cv2.imshow("Pose + Angle ", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()