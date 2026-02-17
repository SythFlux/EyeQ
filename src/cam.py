from flask import Flask, Response
import cv2
import time
from ultralytics import YOLO
import torch

app = Flask(__name__)

DEV = "/dev/video0"

def make_camera():
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    return cap

cap = make_camera()

# ---- CUDA / YOLO setup ----
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "Device:", DEVICE)
if DEVICE.startswith("cuda"):
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolov8n.pt")

def gen():
    target_fps = 15  # inference + encode cost, start conservative
    frame_interval = 1.0 / target_fps
    last = 0.0

    while True:
        now = time.time()
        if now - last < frame_interval:
            time.sleep(0.001)
            continue
        last = now

        ret, frame = cap.read()
        if not ret:
            continue

        # YOLO expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference on GPU
        # imgsz smaller = faster (try 640 or 416)
        results = model.predict(
            source=rgb,
            device=0 if DEVICE.startswith("cuda") else "cpu",
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        # Draw boxes (Ultralytics returns an annotated image via plot())
        annotated_rgb = results[0].plot()  # RGB image (numpy)
        annotated = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/")
def index():
    return """
    <html>
      <head><title>Jetson Live View (YOLO CUDA)</title></head>
      <body style="margin:0;background:#111;display:flex;align-items:center;justify-content:center;height:100vh;">
        <img src="/video" style="max-width:100%;max-height:100vh;" />
      </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
