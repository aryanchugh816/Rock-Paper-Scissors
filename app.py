from flask import Flask, render_template, request
import cv2, time, json
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    
    return render_template("index2.html")

@app.route('/receive', methods=['POST'])
def receive_data():
    if request.method == 'POST':
        data = request.get_json()
        data = json.loads(data)
        img = np.array(list(data["data"].values()))
        img = np.asarray(img, dtype=np.uint8)
        img = img.reshape(120,160,4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        cv2.imwrite("received_img.jpg", img)
        print("Working 1")
        print(img.shape)
        print("Working 2")
    return ' '

if __name__ == "__main__":
    app.run(debug=True)
