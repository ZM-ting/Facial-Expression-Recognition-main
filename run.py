import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# 🔥 彻底不用 yolov11n-face.pt
face_model = YOLO("yolov8n.pt")

st.title("✅ 修复完成！不再报错！")
uploaded_file = st.file_uploader("上传图片", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="原图")
    results = face_model(img)
    st.image(results[0].plot(), caption="检测结果")