import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# --------------------------
# 模型加载（自动下载，不用手动放）
# --------------------------
# 人脸检测模型（官方自带，会自动下载）
face_model = YOLO("yolov8n.pt")  # 官方自带，不会报错

# 人脸检测/考勤用这个
face_model = YOLO("yolov11n-face.pt")

# --------------------------
# 网页界面
# --------------------------
st.set_page_config(page_title="课堂考勤&表情识别", layout="wide")
st.title("🎓 课堂智能考勤 + 表情识别系统")
st.divider()

# 上传图片
uploaded_file = st.file_uploader("上传课堂照片", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # 显示原图
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 原图")
        st.image(image, use_column_width=True)

    # --------------------------
    # 人脸检测
    # --------------------------
    face_results = face_model(img_np, conf=0.4)
    face_count = len(face_results[0].boxes)

    # --------------------------
    # 表情识别
    # --------------------------
    emotion_results = emotion_model(img_np, conf=0.5)

    # --------------------------
    # 绘制结果图
    # --------------------------
    result_img = emotion_results[0].plot()
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    with col2:
        st.subheader("✅ 识别结果")
        st.image(result_img, use_column_width=True)

    st.divider()

    # --------------------------
    # 考勤统计
    # --------------------------
    st.subheader("📊 考勤 & 状态统计")
    st.success(f"**检测到总人数：{face_count} 人**")
    st.info("✅ 出勤完成 | 🎭 表情识别完成")

    # 表情统计
    emotion_names = emotion_results[0].names
    emotion_counts = {}
    for cls in emotion_results[0].boxes.cls:
        name = emotion_names[int(cls)]
        emotion_counts[name] = emotion_counts.get(name, 0) + 1

    st.subheader("😀 学生表情分布")
    for k, v in emotion_counts.items():
        st.write(f"- {k}：{v} 人")

st.divider()
st.caption("✅ 系统说明：本系统用于课堂自动考勤 + 表情状态分析，唯一运行入口")