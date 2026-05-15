import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5)

print("테스트 시작: 손이 인식되면 화면에 큰 원이 그려져야 합니다.")

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # 9번 좌표(손바닥 중앙)에 지름 200짜리 거대한 형광색 원 그리기
            lm = handLms.landmark[9]
            cx, cy = int(lm.x * w), int(lm.y * h)
            
            # ⭐ 이 코드가 실행되면 화면에 엄청나게 큰 원이 보여야 함
            cv2.circle(img, (cx, cy), 100, (0, 255, 255), -1) 
            cv2.putText(img, "WORKING!!!", (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    cv2.imshow("TEST", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()