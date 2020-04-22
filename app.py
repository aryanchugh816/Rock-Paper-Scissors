from flask import Flask, render_template
import cv2, time

app = Flask(__name__)

@app.route('/')
def index():
    print("Working")
    while True:
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            print("pressed")
    #return render_template("index2.html", cap="no")

if __name__ == "__main__":
    app.run(debug=True)
