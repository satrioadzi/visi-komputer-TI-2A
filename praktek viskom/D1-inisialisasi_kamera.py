import cv2, time
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frames, t0 = 0, time.time()
while True:
    ok, frame = cap.read()
    frames= frames + 1
    if time.time() - t0 >= 1.0:
        cv2.setWindowTitle("Kamera", f"Kamera - FPS: {frames}")
        frames, t0 = 0, time.time()
    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()