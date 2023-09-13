from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def my_video():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read(0)
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response('index.html')

@app.route('/video')
def video():
    return Response(my_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)