import cv2
import io
import numpy as np
import json
import tflite_runtime.interpreter as tflite
import redis
import datetime

rd = redis.Redis(host="localhost", port=6379)

labels = ["clean", "messy"]

inter = tflite.Interpreter(
    model_path="model.tflite",
    experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")],
)

inter.allocate_tensors()
cam = cv2.VideoCapture(1)  # ignore the errors
r, frame = cam.read()
img = cv2.resize(frame, (224, 224))

inter.set_tensor(inter.get_input_details()[0]["index"], [img])
inter.invoke()

output_data = inter.get_tensor(inter.get_output_details()[0]["index"])

nn_output = output_data[0]
class_id = np.argmax(nn_output)
score = nn_output[class_id] / np.sum(nn_output)

res = {
    "score": score,
    "label": labels[class_id],
    "last_checked": str(datetime.datetime.today()),
}

rd.set("workbench_status", json.dumps(res))

val = rd.get("workbench_status")

print(val)
