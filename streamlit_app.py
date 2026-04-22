import streamlit as st
import cv2
from ultralytics import YOLO
import os
import pandas as pd
from datetime import datetime, time
import warnings

warnings.filterwarnings("ignore")

# ===================== 全局配置（存储路径已改为项目主文件夹） =====================
# 权重文件路径（项目主文件夹下）
WEIGHT_PATH = "yolov11n-face.pt"
# 考勤基础配置（所有数据存储到项目主文件夹的「考勤记录」子文件夹）
ATTENDANCE_CONFIG = {
    "上班时间": time(9, 0),  # 9:00上班
    "下班时间": time(18, 0),  # 18:00下班
    "迟到阈值": 10,  # 迟到10分钟内算正常（可选）
    "早退阈值": 10,  # 早退10分钟内算正常（可选）
    "考勤记录路径": "考勤记录\\attendance_records.xlsx",  # 项目主文件夹/考勤记录/
    "员工信息路径": "考勤记录\\employee_info.xlsx",  # 项目主文件夹/考勤记录/
    "重复打卡间隔": 600  # 10分钟内禁止重复打卡（秒）
}
# 摄像头配置
CAMERA_ID = 0
CONFIDENCE_THRESHOLD = 0.5

# 页面配置
st.set_page_config(page_title="YOLOv11 人脸考勤系统（最终版）", layout="wide")
st.title("🎯 YOLOv11 人脸考勤系统（最终版）")

# ===================== 初始化状态 =====================
if "stop" not in st.session_state:
    st.session_state.stop = False
if "run" not in st.session_state:
    st.session_state.run = False
if "attendance_data" not in st.session_state:
    st.session_state.attendance_data = []
if "employee_list" not in st.session_state:
    st.session_state.employee_list = []


# ===================== 核心工具函数 =====================
def init_employee_info():
    """初始化员工信息表（确保编号为字符串格式，存储到项目主文件夹）"""
    # 自动创建项目主文件夹下的「考勤记录」子文件夹
    os.makedirs(os.path.dirname(ATTENDANCE_CONFIG["员工信息路径"]), exist_ok=True)
    if not os.path.exists(ATTENDANCE_CONFIG["员工信息路径"]):
        df = pd.DataFrame({
            "员工编号": ["001", "002"],  # 强制字符串格式
            "员工姓名": ["张三", "李四"],
            "部门": ["技术部", "行政部"],
            "岗位": ["开发", "文员"]
        })
        df.to_excel(ATTENDANCE_CONFIG["员工信息路径"], index=False)
    # 读取并更新会话状态（指定员工编号为字符串）
    df = pd.read_excel(ATTENDANCE_CONFIG["员工信息路径"], dtype={"员工编号": str})
    st.session_state.employee_list = df.to_dict("records")


def load_model():
    """加载YOLO人脸检测模型"""
    if not os.path.exists(WEIGHT_PATH):
        st.error(f"❌ 权重文件不存在：{WEIGHT_PATH}")
        return None
    try:
        model = YOLO(WEIGHT_PATH)
        st.success("✅ 模型加载成功！")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败：{str(e)}")
        return None


def check_attendance_rule(check_time, check_type="上班"):
    """考勤规则校验（支持上班/下班）"""
    check_time_obj = check_time.time()
    status = "正常"
    duration = 0

    if check_type == "上班":
        work_start = ATTENDANCE_CONFIG["上班时间"]
        if check_time_obj >= work_start:
            delta = (datetime.combine(datetime.today(), check_time_obj) -
                     datetime.combine(datetime.today(), work_start))
            duration = int(delta.total_seconds() / 60)
            status = f"迟到{duration}分钟" if duration > 0 else "正常"
    else:  # 下班
        work_end = ATTENDANCE_CONFIG["下班时间"]
        if check_time_obj <= work_end:
            delta = (datetime.combine(datetime.today(), work_end) -
                     datetime.combine(datetime.today(), check_time_obj))
            duration = int(delta.total_seconds() / 60)
            status = f"早退{duration}分钟" if duration > 0 else "正常"

    return status, duration


def is_duplicate_check(emp_id):
    """检查是否重复打卡（10分钟内）"""
    if not st.session_state.attendance_data:
        return False
    emp_records = [r for r in st.session_state.attendance_data if r["员工编号"] == emp_id]
    if not emp_records:
        return False
    latest_time = datetime.strptime(emp_records[-1]["打卡时间"], "%Y-%m-%d %H:%M:%S")
    return (datetime.now() - latest_time).total_seconds() < ATTENDANCE_CONFIG["重复打卡间隔"]


def save_attendance(emp_id, emp_name, face_detected=True, check_type="上班"):
    """保存考勤记录（含重复打卡校验，存储到项目主文件夹）"""
    # 重复打卡校验
    if is_duplicate_check(emp_id):
        st.sidebar.error("❌ 10分钟内已打卡，禁止重复操作！")
        return

    now = datetime.now()
    check_status, duration = check_attendance_rule(now, check_type)

    # 构造完整记录
    record = {
        "员工编号": emp_id,
        "员工姓名": emp_name,
        "打卡类型": check_type + "打卡",
        "打卡时间": now.strftime("%Y-%m-%d %H:%M:%S"),
        "人脸检测状态": "✅ 成功" if face_detected else "❌ 失败",
        "考勤状态": check_status if face_detected else "未识别到人脸（缺勤）",
        "打卡日期": now.strftime("%Y-%m-%d"),
        "工时(小时)": round((datetime.combine(now.date(), ATTENDANCE_CONFIG["下班时间"]) -
                             datetime.combine(now.date(), ATTENDANCE_CONFIG["上班时间"])).total_seconds() / 3600, 1)
    }
    st.session_state.attendance_data.append(record)

    # 保存到项目主文件夹的「考勤记录」子文件夹
    try:
        df = pd.DataFrame(st.session_state.attendance_data)
        df.to_excel(ATTENDANCE_CONFIG["考勤记录路径"], index=False, encoding="utf-8")
        # 显示存储路径（方便确认）
        st.sidebar.success(f"📝 考勤记录已保存：{os.path.abspath(ATTENDANCE_CONFIG['考勤记录路径'])}")
    except Exception as e:
        st.sidebar.error(f"❌ 保存失败：{str(e)}")


def add_new_employee(emp_id, emp_name, dept, position="未知"):
    """新增员工信息（存储到项目主文件夹）"""
    if not emp_id or not emp_name:
        return False, "编号和姓名不能为空！"

    # 读取现有数据（项目主文件夹下的员工表）
    df = pd.read_excel(ATTENDANCE_CONFIG["员工信息路径"], dtype={"员工编号": str})
    # 检查编号是否重复
    if emp_id in df["员工编号"].values:
        return False, f"员工编号{emp_id}已存在！"

    # 新增行
    new_row = pd.DataFrame({
        "员工编号": [emp_id],
        "员工姓名": [emp_name],
        "部门": [dept],
        "岗位": [position]
    })
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(ATTENDANCE_CONFIG["员工信息路径"], index=False)
    # 刷新会话状态
    st.session_state.employee_list = df.to_dict("records")
    return True, f"新增员工{emp_name}（{emp_id}）成功！"


def generate_attendance_report():
    """生成考勤报表（支持日期筛选+异常标红，存储到项目主文件夹）"""
    if not st.session_state.attendance_data:
        st.warning("⚠️ 暂无考勤数据，无法生成报表")
        return

    # 日期筛选
    st.subheader("📅 报表日期筛选")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", datetime.today())
    with col2:
        end_date = st.date_input("结束日期", datetime.today())

    # 数据处理
    df = pd.DataFrame(st.session_state.attendance_data)
    df["打卡日期"] = pd.to_datetime(df["打卡日期"])
    df["打卡时间"] = pd.to_datetime(df["打卡时间"])
    # 筛选指定日期范围
    df_filtered = df[(df["打卡日期"] >= pd.to_datetime(start_date)) &
                     (df["打卡日期"] <= pd.to_datetime(end_date))]

    if df_filtered.empty:
        st.warning("⚠️ 所选日期范围内无考勤数据！")
        return

    # 1. 生成汇总报表（异常标红）
    st.subheader("📊 考勤汇总报表")
    status_counts = df_filtered.groupby(["员工姓名", "考勤状态"]).size().unstack(fill_value=0)
    status_counts["打卡总次数"] = status_counts.sum(axis=1)

    # 异常状态标红
    def highlight_abnormal(val):
        color = 'red' if "迟到" in str(val) or "早退" in str(val) or "缺勤" in str(val) else 'black'
        return f'color: {color}'

    styled_df = status_counts.style.applymap(highlight_abnormal)
    st.dataframe(styled_df, use_container_width=True)

    # 2. 导出多格式报表（存储到项目主文件夹的「考勤记录」子文件夹）
    st.subheader("💾 导出报表")
    # 构建报表文件名（带日期，避免覆盖）
    base_name = os.path.splitext(ATTENDANCE_CONFIG["考勤记录路径"])[0]
    base_path = f"{base_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    # Excel导出（项目主文件夹/考勤记录/）
    excel_path = f"{base_path}_汇总报表.xlsx"
    df_filtered.to_excel(excel_path, index=False)
    st.success(f"✅ Excel报表：{os.path.abspath(excel_path)}")

    # CSV导出（兼容更多软件，项目主文件夹/考勤记录/）
    csv_path = f"{base_path}_汇总报表.csv"
    df_filtered.to_csv(csv_path, index=False, encoding="utf-8-sig")
    st.success(f"✅ CSV报表：{os.path.abspath(csv_path)}")

    # 3. 测试用例验证
    st.subheader("🧪 测试用例验证")
    test_check_time = datetime(2024, 5, 20, 9, 10)
    test_status, test_duration = check_attendance_rule(test_check_time)
    st.write(f"测试用例：9:10打卡（上班时间9:00）→ 判定结果：{test_status}")
    if test_status == "迟到10分钟":
        st.success("✅ 测试用例验证通过！")
    else:
        st.error(f"❌ 测试用例验证失败，预期：迟到10分钟，实际：{test_status}")


# ===================== 页面交互逻辑 =====================
# 1. 初始化员工信息（存储到项目主文件夹）
init_employee_info()

# 2. 加载模型
model = load_model()

# 3. 侧边栏：核心功能区
st.sidebar.title("👤 员工打卡管理")

# 3.1 员工选择（显示001-张三，而非1-张三）
emp_options = [f"{emp['员工编号']}-{emp['员工姓名']}" for emp in st.session_state.employee_list]
selected_emp = st.sidebar.selectbox("选择员工", emp_options)
emp_id, emp_name = selected_emp.split("-")

# 3.2 打卡类型选择（上班/下班）
check_type = st.sidebar.radio("打卡类型", ["上班", "下班"], horizontal=True)

# 3.3 手动录入打卡
st.sidebar.subheader("📝 手动录入打卡")
manual_check_time = st.sidebar.time_input("打卡时间", value=datetime.now().time())
if st.sidebar.button("手动提交打卡", type="primary"):
    manual_datetime = datetime.combine(datetime.today(), manual_check_time)
    manual_status, _ = check_attendance_rule(manual_datetime, check_type)

    # 重复打卡校验
    if is_duplicate_check(emp_id):
        st.sidebar.error("❌ 10分钟内已打卡，禁止重复操作！")
    else:
        # 构造手动打卡记录
        record = {
            "员工编号": emp_id,
            "员工姓名": emp_name,
            "打卡类型": check_type + "打卡",
            "打卡时间": manual_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "人脸检测状态": "✅ 手动打卡",
            "考勤状态": manual_status,
            "打卡日期": manual_datetime.strftime("%Y-%m-%d"),
            "工时(小时)": round((datetime.combine(manual_datetime.date(), ATTENDANCE_CONFIG["下班时间"]) -
                                 datetime.combine(manual_datetime.date(),
                                                  ATTENDANCE_CONFIG["上班时间"])).total_seconds() / 3600, 1)
        }
        st.session_state.attendance_data.append(record)
        # 保存到项目主文件夹的「考勤记录」子文件夹
        pd.DataFrame(st.session_state.attendance_data).to_excel(ATTENDANCE_CONFIG["考勤记录路径"], index=False)
        st.sidebar.success(f"✅ 手动打卡记录已保存：{os.path.abspath(ATTENDANCE_CONFIG['考勤记录路径'])}")

# 3.4 员工信息管理（新增员工，存储到项目主文件夹）
st.sidebar.markdown("---")
st.sidebar.title("👥 员工信息管理")
new_emp_id = st.sidebar.text_input("新增员工编号（如003）")
new_emp_name = st.sidebar.text_input("新增员工姓名")
new_emp_dept = st.sidebar.text_input("所属部门")
if st.sidebar.button("添加新员工"):
    success, msg = add_new_employee(new_emp_id, new_emp_name, new_emp_dept)
    if success:
        st.sidebar.success(f"✅ {msg}")
    else:
        st.sidebar.error(f"❌ {msg}")

# 3.5 打卡历史记录（优化显示，显示完整存储路径）
st.sidebar.markdown("---")
st.sidebar.title("📋 最新打卡记录")
if st.session_state.attendance_data:
    history_df = pd.DataFrame(st.session_state.attendance_data)[
        ["员工编号", "员工姓名", "打卡类型", "打卡时间", "考勤状态"]
    ].tail(10)  # 显示最新10条
    # 优化时间格式
    history_df["打卡时间"] = pd.to_datetime(history_df["打卡时间"]).dt.strftime("%Y-%m-%d %H:%M")
    st.sidebar.dataframe(history_df, use_container_width=True)
    # 显示数据存储位置
    st.sidebar.info(f"📂 数据存储位置：{os.path.abspath('考勤记录')}")
else:
    st.sidebar.info("暂无打卡记录")

# 4. 主界面：摄像头打卡
if model is not None:
    st.subheader("📹 摄像头实时打卡")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 启动摄像头打卡", type="primary", use_container_width=True):
            st.session_state.run = True
            st.session_state.stop = False
    with col2:
        if st.button("🛑 停止摄像头", use_container_width=True):
            st.session_state.stop = True
            st.session_state.run = False

    # 摄像头画面展示
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    # 摄像头核心逻辑
    cap = None
    is_saved = False
    try:
        if st.session_state.run and not st.session_state.stop:
            cap = cv2.VideoCapture(CAMERA_ID)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            while cap.isOpened() and not st.session_state.stop:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.warning("⚠️ 摄像头读取失败，请检查：1. 摄像头是否被占用 2. 摄像头编号是否正确")
                    break

                # 人脸检测
                results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                frame = results[0].plot()  # 绘制检测框
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检测到人脸且未保存记录时，执行打卡
                face_count = len(results[0].boxes)
                if face_count > 0 and not is_saved:
                    status_placeholder.success(f"✅ 检测到{face_count}个人脸，正在保存{check_type}打卡记录...")
                    save_attendance(emp_id, emp_name, face_detected=True, check_type=check_type)
                    is_saved = True
                elif face_count == 0:
                    status_placeholder.warning("⚠️ 未检测到人脸，请正对摄像头")

                # 显示画面
                frame_placeholder.image(frame_rgb, channels="RGB", width=640)

        # 停止摄像头释放资源
        if st.session_state.stop or not st.session_state.run:
            if cap is not None:
                cap.release()
                frame_placeholder.empty()
                status_placeholder.empty()
                st.info("ℹ️ 摄像头已停止")

    except Exception as e:
        st.error(f"❌ 摄像头运行出错：{str(e)}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

# 5. 考勤报表生成
st.markdown("---")
if st.button("📈 生成考勤报表", type="primary", use_container_width=True):
    generate_attendance_report()

# 6. 底部说明（标注存储路径）
st.markdown("---")
st.markdown("### 📢 使用说明")
st.markdown(f"1. 所有数据默认存储到：`{os.path.abspath('考勤记录')}`（项目主文件夹下）")
st.markdown("2. 支持上班/下班两种打卡类型，自动校验迟到/早退规则")
st.markdown("3. 10分钟内禁止重复打卡，避免误操作")
st.markdown("4. 报表支持Excel/CSV格式导出，异常状态自动标红")