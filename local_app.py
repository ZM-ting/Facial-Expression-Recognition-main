import cv2
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov11n-face.pt")
# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("考勤系统启动，按Q退出！")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 人脸检测
    results = model(frame, verbose=False)
    frame = results[0].plot()
    # 显示窗口
    cv2.imshow("课堂人脸考勤系统", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()