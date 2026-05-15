import cv2
import mediapipe as mp
import random
import math

# ------------------ 카메라 ------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ------------------ MediaPipe ------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

particles = []

# ------------------ 이미지 폴더 ------------------
IMG_PATH = r"C:\Users\Yuki\Desktop\hand_tracking_project\image"

# ------------------ PNG 로드 ------------------
fire_img = cv2.imread(
    IMG_PATH + r"\fire.png",
    cv2.IMREAD_UNCHANGED
)

heart_img = cv2.imread(
    IMG_PATH + r"\heart.png",
    cv2.IMREAD_UNCHANGED
)

thanos_img = cv2.imread(
    IMG_PATH + r"\thanos.png",
    cv2.IMREAD_UNCHANGED
)

nason_img = cv2.imread(
    IMG_PATH + r"\nason.png",
    cv2.IMREAD_UNCHANGED
)

# ------------------ resize ------------------
if fire_img is not None:
    fire_img = cv2.resize(fire_img, (50, 50))

if heart_img is not None:
    heart_img = cv2.resize(heart_img, (120, 120))

if thanos_img is not None:
    thanos_img = cv2.resize(thanos_img, (220, 220))

if nason_img is not None:
    nason_img = cv2.resize(nason_img, (180, 180))

# ------------------ 이미지 합성 ------------------
def overlay_image(bg, overlay, x, y):

    if overlay is None:
        return

    h, w = overlay.shape[:2]

    if x < 0 or y < 0:
        return

    if x + w > bg.shape[1] or y + h > bg.shape[0]:
        return

    # 투명 PNG
    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):

            bg[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c] +
                (1 - alpha) * bg[y:y+h, x:x+w, c]
            )

    # 일반 이미지
    else:

        bg[y:y+h, x:x+w] = overlay[:, :, :3]

# ------------------ 불꽃 생성 ------------------
def create_particle(x, y, effect="fire"):

    return {
        "x": x,
        "y": y,
        "vx": random.uniform(-3, 3),
        "vy": random.uniform(-5, -1),
        "life": random.randint(8, 15),
        "effect": effect
    }

# ------------------ 메인 루프 ------------------
while True:

    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            lm = hand.landmark

            # ------------------ 좌표 ------------------
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)

            tx = int(lm[4].x * w)
            ty = int(lm[4].y * h)

            mx = int(lm[12].x * w)
            my = int(lm[12].y * h)

            wx = int(lm[0].x * w)
            wy = int(lm[0].y * h)

            # ------------------ 검지만 ------------------
            one_finger = (
                lm[8].y < lm[6].y and
                lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and
                lm[20].y > lm[18].y
            )

            # ------------------ 주먹 ------------------
            fist = (
                lm[8].y > lm[6].y and
                lm[12].y > lm[10].y and
                lm[16].y > lm[14].y and
                lm[20].y > lm[18].y
            )

            # ------------------ 하트 포즈 ------------------
            distance = math.hypot(ix - tx, iy - ty)

            heart_pose = distance < 40

            # ------------------ 나선환 포즈 ------------------
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            fingers_cross = abs(ix - mx) < 25

            nason_pose = (
                index_up and
                middle_up and
                fingers_cross
            )

            # ------------------ 불꽃 ------------------
            if one_finger:

                for _ in range(2):
                    particles.append(
                        create_particle(ix, iy, "fire")
                    )

            # ------------------ 하트 ------------------
            if heart_pose:

                overlay_image(
                    img,
                    heart_img,
                    ix - 60,
                    iy - 100
                )

            # ------------------ 타노스 ------------------
            if fist:

                overlay_image(
                    img,
                    thanos_img,
                    wx - 110,
                    wy - 110
                )

                for _ in range(3):
                    particles.append(
                        create_particle(wx, wy, "fire")
                    )

            # ------------------ 나선환 ------------------
            if nason_pose:

                center_x = (ix + mx) // 2
                center_y = (iy + my) // 2

                overlay_image(
                    img,
                    nason_img,
                    center_x - 90,
                    center_y - 90
                )

                for _ in range(2):
                    particles.append(
                        create_particle(
                            center_x,
                            center_y,
                            "nason"
                        )
                    )

    # ------------------ 파티클 ------------------
    new_particles = []

    for p in particles:

        p["x"] += p["vx"]
        p["y"] += p["vy"]
        p["life"] -= 1

        if p["life"] > 0:

            if p["effect"] == "fire":

                overlay_image(
                    img,
                    fire_img,
                    int(p["x"]) - 25,
                    int(p["y"]) - 25
                )

            elif p["effect"] == "nason":

                overlay_image(
                    img,
                    nason_img,
                    int(p["x"]) - 40,
                    int(p["y"]) - 40
                )

            new_particles.append(p)

    particles = new_particles

    # ------------------ 출력 ------------------
    cv2.imshow("Hand Effect", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ 종료 ------------------
cap.release()
cv2.destroyAllWindows()