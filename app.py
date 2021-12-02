import cv2
import io
import numpy as np
import tflite_runtime.interpreter as tflite
import datetime

from flask_cors import CORS
from flask import Flask, send_file, jsonify

app = Flask(__name__)
CORS(app)

labels = ["clean", "messy"]

inter = tflite.Interpreter(
            model_path="model.tflite",
            experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
        )
inter.allocate_tensors()


CACHE_MINUTES = 5
cache_status = "unknown"
cache_date = datetime.datetime.today()

@app.route('/')
def index():
    return "test"

@app.route('/status')
def status():
    global cache_status
    global cache_date

    if cache_status == "unknown" or cache_date < datetime.datetime.today():
        cam = cv2.VideoCapture(1)  #ignore the errors
        r, frame = cam.read()
        img = cv2.resize(frame, (224, 224))
        #s, arr = cv2.imencode(".jpg", frame) 
        
        inter.set_tensor(inter.get_input_details()[0]["index"], [img])
        inter.invoke()

        output_data = inter.get_tensor(inter.get_output_details()[0]["index"])

        nn_output = output_data[0]
        class_id = np.argmax(nn_output)
        score = nn_output[class_id] / np.sum(nn_output)

        cache_date = datetime.datetime.today() + datetime.timedelta(minutes=CACHE_MINUTES)

        res = {
            "score": score,
            "label": labels[class_id],
            "last_checked": datetime.datetime.today(),
        }

        cache_status = res
        print("cache expires {}".format(cache_date))

        return jsonify(res)
    else:
        print("returning cached")
        return jsonify(cache_status)

@app.route('/frame.jpg')
def frame():
    cam = cv2.VideoCapture(1)  #ignore the errors
    r, frame = cam.read()

    s, arr = cv2.imencode(".jpg", frame) 

    return send_file(
            io.BytesIO(arr),
            mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host="0.0.0.0")

