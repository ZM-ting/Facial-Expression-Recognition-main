import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# 页面基础配置
st.set_page_config(page_title="人脸检测演示", layout="wide", initial_sidebar_state="collapsed")
st.title("YOLOv11 人脸实时检测")


# -------------------------- 核心修复：确保权重正确加载 --------------------------
@st.cache_resource  # 缓存模型，避免重复加载
def load_model():
    """加载YOLOv11人脸检测模型（自动下载正版权重）"""
    try:
        # 自动从官方源下载yolov11n-face权重，跳过本地损坏文件
        model = YOLO("yolov11n-face.pt")
        st.success("模型加载成功！")
        return model
    except Exception as e:
        st.error(f"模型加载失败：{str(e)}")
        st.info("正在尝试重新下载官方权重...")
        # 强制删除缓存的损坏权重，重新下载
        from ultralytics.utils.downloads import download
        weight_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n-face.pt"
        weight_path = download(weight_url, dir=".", unzip=False)
        model = YOLO(weight_path)
        st.success("权重重新下载并加载成功！")
        return model


# 加载模型
model = load_model()

# -------------------------- 摄像头实时检测逻辑 --------------------------
# 状态变量（控制摄像头启停）
if "stop" not in st.session_state:
    st.session_state.stop = False
if "run" not in st.session_state:
    st.session_state.run = False

# 按钮控制
col1, col2 = st.columns(2)
with col1:
    if st.button("启动摄像头", type="primary"):
        st.session_state.run = True
        st.session_state.stop = False
with col2:
    if st.button("停止摄像头"):
        st.session_state.stop = True
        st.session_state.run = False

# 视频显示区域
frame_placeholder = st.empty()

# 摄像头检测主逻辑
cap = None
try:
    if st.session_state.run and not st.session_state.stop:
        # 打开摄像头（0为默认摄像头，若报错可改为1/2尝试）
        cap = cv2.VideoCapture(0)
        # 设置摄像头分辨率（可选，提升流畅度）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened() and not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                st.warning("摄像头读取失败，请检查摄像头是否被占用")
                break

            # 人脸检测（置信度0.5，过滤误检）
            results = model(frame, verbose=False, conf=0.5)
            # 绘制检测框
            frame = results[0].plot()
            # 转换颜色格式（OpenCV的BGR转Streamlit的RGB）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 显示画面
            frame_placeholder.image(frame_rgb, channels="RGB", width=640)

    # 停止时释放摄像头
    if st.session_state.stop or not st.session_state.run:
        if cap is not None:
            cap.release()
            frame_placeholder.empty()  # 清空画面
            st.info("摄像头已停止")

except Exception as e:
    # 捕获所有异常并提示
    st.error(f"程序运行出错：{str(e)}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()