from flask import Flask, render_template, Response
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        alert(text="Couldn't Open Web Cam", title="Error", button="OK")
        exit()

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()

        if not ret:
            alert(text="Can't receive frame (stream end?). Exiting ...", title="Error", button="OK")
            exit()
            
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.flip(img, 1)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        plt.pause(0.1)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=True)
