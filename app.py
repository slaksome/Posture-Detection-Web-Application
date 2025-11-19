from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math
import threading

app = Flask(__name__)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = None

posture_data = {
    "good_frames": 0,
    "bad_frames": 0,
    "running": False
}

lock = threading.Lock()

def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    ba_mag = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    bc_mag = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if ba_mag == 0 or bc_mag == 0:
        return 0.0
    cosine_angle = (ba[0] * bc[0] + ba[1] * bc[1]) / (ba_mag * bc_mag + 1e-6)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

def posture_check(landmarks, image):
    h, w, _ = image.shape

    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

    rs = (right_shoulder.x * w, right_shoulder.y * h)
    re = (right_ear.x * w, right_ear.y * h)
    rh = (right_hip.x * w, right_hip.y * h)
    ls = (left_shoulder.x * w, left_shoulder.y * h)
    lh = (left_hip.x * w, left_hip.y * h)

    neck_angle = calculate_angle(re, rs, rh)
    shoulder_diff = abs(rs[1] - ls[1])

    slouch = False
    warnings = []

    if neck_angle < 65:
        slouch = True
        warnings.append("Neck bent forward")
    if shoulder_diff > 20:
        slouch = True
        warnings.append("Shoulders uneven")

    cv2.putText(image, f'Neck Angle: {int(neck_angle)}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f'Shoulder Diff: {int(shoulder_diff)}', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    y0 = 130
    for wng in warnings:
        cv2.putText(image, 'Warning: ' + wng, (30, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        y0 += 40

    return slouch

def generate_frames():
    global cap

    while True:
        with lock:
            if not posture_data["running"] or cap is None:
                break

        success, frame = cap.read()
        if not success:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        slouch = False
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            slouch = posture_check(results.pose_landmarks.landmark, image)

        with lock:
            if slouch:
                posture_data["bad_frames"] += 1
            else:
                posture_data["good_frames"] += 1

        ret, buffer = cv2.imencode('.jpg', image)
        if not ret:
            continue

        frame = buffer.tobytes()

        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

    with lock:
        if cap is not None:
            cap.release()
            cap = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    with lock:
        if cap is None or not posture_data["running"]:
            return "Camera not started", 404
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global cap
    with lock:
        if not posture_data["running"]:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = None
                return jsonify({"status": "error", "message": "Cannot open camera"}), 500
            posture_data["good_frames"] = 0
            posture_data["bad_frames"] = 0
            posture_data["running"] = True
    return jsonify({"status": "started"})

@app.route('/stop', methods=['POST'])
def stop():
    global cap
    with lock:
        posture_data["running"] = False
        total = posture_data["good_frames"] + posture_data["bad_frames"]
        good_pct = (posture_data["good_frames"] / total * 100) if total > 0 else 0
        bad_pct = (posture_data["bad_frames"] / total * 100) if total > 0 else 0
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None

    return jsonify({
        "status": "stopped",
        "good_frames": posture_data["good_frames"],
        "bad_frames": posture_data["bad_frames"],
        "good_pct": round(good_pct, 2),
        "bad_pct": round(bad_pct, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)