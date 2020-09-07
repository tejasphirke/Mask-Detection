
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        #count1()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    #return render_template('index.html',gen(VideoCamera()),count=count)
    
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/count1')
def count1():
    f = open("countfile.txt", "r")
    #   print("read value is "+f.read())
    a = f.read()
    print("a is "+a)
    return a

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
