from flask import Flask, render_template, request
import cv2,json
import numpy as np
from functions import *
import tensorflow as tf
global graph, model

graph = tf.get_default_graph()

model = create_model()

app = Flask(__name__)

@app.route('/')
def index():

    return render_template("index2.html")

@app.route('/receive', methods=['POST'])
def receive_data():
    labels = {0: 'Paper', 2: 'Scissor', 1: 'Rock'}
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data)
        img = np.array(list(data["data"].values()))
        img = np.asarray(img, dtype=np.uint8)
        img = img.reshape(120,160,4)
        batch = img_preprocess(img)
        cv2.imwrite("received_img.jpg", img)
        with graph.as_default():
            res = model.predict(batch)
            res = labels[np.argmax(res[0])]
        print("Working 1")
        print(res)
        print("Working 2")
        res = {"result": res}
        res = json.dumps(res)
        print(type(res))
    return res

if __name__ == "__main__":
    app.run(debug=True)
