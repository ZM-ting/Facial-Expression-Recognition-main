import streamlit as st
import cv2
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime

# ===================== 全局配置（可根据自己需求修改） =====================
# 权重文件路径（确保yolov11n-face.pt已放到项目根目录）
WEIGHT_PATH = "yolov11n-face.pt"
# 考勤记录保存路径（建议改成桌面/易找到的位置）
ATTENDANCE_SAVE_PATH = "D:\\考勤记录\\attendance_records.xlsx"
# 摄像头编号（0=默认摄像头，打不开改1/2）
CAMERA_ID = 0
# 人脸检测置信度（0.5=过滤50%以下的误检）
CONFIDENCE_THRESHOLD = 0.5

# 页面基础配置
st.set_page_config(
    page_title="YOLOv11 人脸考勤系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("🎯 YOLOv11 人脸实时检测 + 考勤记录系统")

# ===================== 初始化全局状态 =====================
# 摄像头控制状态
if "stop" not in st.session_state:
    st.session_state.stop = False
if "run" not in st.session_state:
    st.session_state.run = False
# 考勤数据存储（内存中临时保存）
if "attendance_data" not in st.session_state:
    st.session_state.attendance_data = []


# ===================== 核心功能函数 =====================
@st.cache_resource  # 缓存模型，避免重复加载
def load_model():
    """加载本地YOLOv11人脸检测模型（离线版）"""
    # 检查权重文件是否存在
    if not os.path.exists(WEIGHT_PATH):
        st.error(f"❌ 未找到权重文件！请确认 {WEIGHT_PATH} 已放到项目根目录")
        st.info("✅ 权重文件下载地址：https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov11n-face.pt")
        return None

    # 加载本地模型
    try:
        model = YOLO(WEIGHT_PATH)
        st.success("✅ 模型加载成功！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        st.info("💡 可能原因：权重文件损坏/版本不匹配，请重新下载正版权重")
        return None


def save_attendance(face_detected=True, custom_note=""):
    """
    保存考勤打卡记录到Excel
    :param face_detected: 是否检测到人脸
    :param custom_note: 自定义备注（可选）
    """
    # 构造打卡记录
    record = {
        "打卡时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "人脸检测状态": "✅ 成功" if face_detected else "❌ 失败",
        "考勤结果": "正常打卡" if face_detected else "未识别到人脸",
        "备注": custom_note
    }
    st.session_state.attendance_data.append(record)

    # 确保保存文件夹存在
    os.makedirs(os.path.dirname(ATTENDANCE_SAVE_PATH), exist_ok=True)

    # 写入Excel（覆盖式保存，保留所有历史记录）
    try:
        df = pd.DataFrame(st.session_state.attendance_data)
        df.to_excel(ATTENDANCE_SAVE_PATH, index=False)
        st.sidebar.success(f"📝 考勤记录已保存到：\n{ATTENDANCE_SAVE_PATH}")
    except Exception as e:
        st.sidebar.error(f"❌ 保存考勤记录失败：{str(e)}")


def show_attendance_history():
    """在侧边栏显示考勤历史记录"""
    st.sidebar.title("📊 考勤历史记录")
    if st.session_state.attendance_data:
        # 显示最新10条记录
        latest_records = st.session_state.attendance_data[-10:]
        df_latest = pd.DataFrame(latest_records)
        st.sidebar.dataframe(df_latest, use_container_width=True)

        # 导出全部记录按钮
        if st.sidebar.button("📥 导出全部考勤记录"):
            save_attendance()  # 触发一次保存，确保最新数据写入
            st.sidebar.success("✅ 全部记录已导出到指定路径！")
    else:
        st.sidebar.info("暂无考勤记录，请启动摄像头打卡～")


# ===================== 主程序逻辑 =====================
# 加载模型
model = load_model()

# 只有模型加载成功才显示核心功能
if model is not None:
    # 显示考勤记录侧边栏
    show_attendance_history()

    # 摄像头控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 启动摄像头/打卡", type="primary", use_container_width=True):
            st.session_state.run = True
            st.session_state.stop = False
    with col2:
        if st.button("🛑 停止摄像头", use_container_width=True):
            st.session_state.stop = True
            st.session_state.run = False

    # 视频显示区域
    frame_placeholder = st.empty()
    # 检测状态提示
    status_placeholder = st.empty()

    # 摄像头检测主逻辑
    cap = None
    try:
        if st.session_state.run and not st.session_state.stop:
            # 打开摄像头
            cap = cv2.VideoCapture(CAMERA_ID)
            # 设置摄像头分辨率（提升流畅度）
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # 标记是否已保存本次打卡记录（避免重复保存）
            is_saved = False

            while cap.isOpened() and not st.session_state.stop:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.warning("⚠️ 摄像头读取失败，请检查：\n1. 摄像头是否被占用\n2. 摄像头编号是否正确")
                    break

                # 人脸检测
                results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                # 绘制检测框
                frame = results[0].plot()
                # 转换颜色格式（OpenCV BGR → Streamlit RGB）
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测人脸数量并更新状态
                face_count = len(results[0].boxes)
                if face_count > 0:
                    status_placeholder.success(f"✅ 检测到 {face_count} 个人脸 | 置信度：{CONFIDENCE_THRESHOLD}")
                    # 仅首次检测到人脸时保存考勤记录（避免每秒重复保存）
                    if not is_saved:
                        save_attendance(face_detected=True, custom_note=f"检测到{face_count}个人脸")
                        is_saved = True
                else:
                    status_placeholder.warning("⚠️ 未检测到人脸，请正对摄像头")
                    # 未检测到人脸时，仅首次标记保存
                    if not is_saved:
                        save_attendance(face_detected=False, custom_note="未识别到人脸")
                        is_saved = True

                # 显示摄像头画面
                frame_placeholder.image(frame_rgb, channels="RGB", width=640)

        # 停止摄像头时释放资源
        if st.session_state.stop or not st.session_state.run:
            if cap is not None:
                cap.release()
                frame_placeholder.empty()  # 清空画面
                status_placeholder.empty()  # 清空状态提示
                st.info("ℹ️ 摄像头已停止 | 最后一次打卡记录已保存")

    except Exception as e:
        # 全局异常捕获，避免程序崩溃
        st.error(f"❌ 程序运行出错：{str(e)}")
        status_placeholder.error(f"❌ 错误详情：{str(e)}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        # 保存异常记录
        save_attendance(face_detected=False, custom_note=f"程序异常：{str(e)}")

# ===================== 底部提示 =====================
st.markdown("---")
st.markdown("### 📢 使用说明")
st.markdown("1. 启动摄像头后，请正对镜头，系统自动检测人脸并保存打卡记录")
st.markdown(f"2. 考勤记录默认保存到：`{ATTENDANCE_SAVE_PATH}`（可在代码顶部修改路径）")
st.markdown("3. 若摄像头无法打开，尝试修改代码顶部的 CAMERA_ID（0→1→2）")