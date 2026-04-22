from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import os

# 初始化Flask应用（去掉CORS，本地使用无需跨域）
app = Flask(__name__)

# 加载你的YOLO人脸权重
model = YOLO("yolov11n-face.pt")  # 确保这个文件在同目录
camera = None


# 生成摄像头帧（人脸检测核心）
def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)  # 0是默认摄像头
    while True:
        success, frame = camera.read()
        if not success:
            break
        # YOLO人脸检测+绘制框
        results = model(frame, verbose=False)
        frame = results[0].plot()

        # 转换为网页可显示的格式
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 主页（自动生成网页界面）
@app.route('/')
def index():
    # 自动创建templates文件夹和index.html，不用手动写
    if not os.path.exists('templates'):
        os.makedirs('templates')
        with open('templates/index.html', 'w', encoding='utf-8') as f:
            f.write('''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>课堂人脸考勤系统</title>
    <style>
        body {text-align: center; font-family: 微软雅黑; margin: 20px;}
        h1 {color: #333;}
        .video {width: 80%; margin: 0 auto; border: 3px solid #009688; border-radius: 10px;}
    </style>
</head>
<body>
    <h1>🧑‍🏫 课堂人脸考勤系统（网页版）</h1>
    <div class="video">
        <img src="/video_feed" alt="实时人脸检测">
    </div>
</body>
</html>
            ''')
    return render_template('index.html')


# 摄像头视频流接口
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# 启动网页
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)