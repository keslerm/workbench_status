import cv2
import io
import redis

from flask_cors import CORS
from flask import Flask, send_file, jsonify

app = Flask(__name__)
CORS(app)

rd = redis.Redis(host="localhost", port=6379)


@app.route("/")
def index():
    return "test"


@app.route("/status")
def status():
    val = rd.get("workbench_status")
    return val


@app.route("/frame.jpg")
def frame():
    cam = cv2.VideoCapture(1)  # ignore the errors
    r, frame = cam.read()

    s, arr = cv2.imencode(".jpg", frame)

    return send_file(io.BytesIO(arr), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
