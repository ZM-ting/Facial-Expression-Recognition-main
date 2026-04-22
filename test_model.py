from ultralytics import YOLO
import cv2

# 加载模型（自动下载）
face_model = YOLO("yolov8n.pt")
emotion_model = YOLO("best.pt")

# 测试图片
img = cv2.imread("test.jpg")

# 检测
face_res = face_model(img)
emotion_res = emotion_model(img)

# 输出
print("="*50)
print("人脸检测完成")
print(f"检测到人数：{len(face_res[0].boxes)}")
print("表情识别完成")
print("="*50)

cv2.imwrite("result.jpg", emotion_res[0].plot())
print("结果已保存为 result.jpg")