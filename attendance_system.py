import cv2
import dlib
import face_recognition
import os
import time
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import shutil

# ===================== 配置项 =====================
FACE_DET_MODEL_PATH = "yolov11n-face.pt"  # 你的人脸检测权重
SHAPE_PREDICTOR_PATH = "model_data/shape_predictor_68_face_landmarks.dat"
FACE_REC_MODEL_PATH = "model_data/dlib_face_recognition_resnet_model_v1.dat"
STUDENT_DB_PATH = "dataset/students"  # 学生人脸库路径
REPORT_SAVE_PATH = "attendance_reports"  # 考勤报表保存路径
TOLERANCE = 0.6  # 人脸匹配阈值（越小越严格）
CAMERA_ID = 0  # 摄像头ID（0为默认摄像头）

# ===================== 初始化 =====================
# 创建报表保存目录
os.makedirs(REPORT_SAVE_PATH, exist_ok=True)

# 加载模型
print("加载模型中...")
face_det_model = YOLO(FACE_DET_MODEL_PATH)
shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
face_rec_model = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)


# ===================== 核心函数 =====================
def load_student_database():
    """加载学生人脸库，生成姓名-学号-特征映射"""
    student_db = {}
    if not os.path.exists(STUDENT_DB_PATH):
        print(f"学生人脸库目录 {STUDENT_DB_PATH} 不存在！")
        return student_db

    for student_folder in os.listdir(STUDENT_DB_PATH):
        folder_path = os.path.join(STUDENT_DB_PATH, student_folder)
        if not os.path.isdir(folder_path):
            continue

        # 解析姓名和学号（文件夹名格式：张三_2023001）
        try:
            name, student_id = student_folder.split("_")
        except:
            print(f"文件夹名格式错误：{student_folder}，请改为「姓名_学号」")
            continue

        # 提取该学生所有照片的人脸特征
        face_encodings = []
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            # 加载图片并提取人脸特征
            img = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                face_encodings.append(encodings[0])

        if face_encodings:
            student_db[student_id] = {
                "name": name,
                "encodings": face_encodings,
                "attendance": "未签到",
                "sign_time": ""
            }
            print(f"加载学生：{name}（学号：{student_id}），特征数：{len(face_encodings)}")
        else:
            print(f"未提取到 {name} 的人脸特征，请检查照片！")

    print(f"\n学生库加载完成，共加载 {len(student_db)} 名学生\n")
    return student_db


def create_attendance_report(student_db, class_name="计算机班", date=None):
    """生成考勤报表（CSV格式）"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    report_filename = f"{class_name}_考勤报表_{date}.csv"
    report_path = os.path.join(REPORT_SAVE_PATH, report_filename)

    # 整理报表数据
    report_data = []
    for student_id, info in student_db.items():
        report_data.append({
            "学号": student_id,
            "姓名": info["name"],
            "考勤状态": info["attendance"],
            "签到时间": info["sign_time"],
            "考勤日期": date,
            "班级": class_name
        })

    # 保存CSV（支持中文）
    df = pd.DataFrame(report_data)
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"考勤报表已保存：{report_path}")
    return report_path


def run_attendance_system(student_db, class_name="计算机班"):
    """运行实时考勤系统"""
    if not student_db:
        print("学生库为空，无法启动考勤！")
        return

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("摄像头打开失败！请检查摄像头是否被占用")
        return

    # 已签到学生集合（避免重复签到）
    signed_students = set()
    # 考勤开始时间
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"课堂考勤启动！开始时间：{start_time}")
    print("操作说明：按 S 保存报表 | 按 Q 退出考勤\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败！")
            break

        # 1. YOLO人脸检测（定位人脸位置）
        results = face_det_model(frame, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 裁剪人脸区域
                face_roi = frame[y1:y2, x1:x2]

                # 2. 提取当前人脸特征
                face_encodings = face_recognition.face_encodings(face_roi)
                if not face_encodings:
                    # 未提取到特征（陌生人/遮挡）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "未知人员", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    continue

                current_encoding = face_encodings[0]
                matched = False

                # 3. 匹配学生库
                for student_id, info in student_db.items():
                    matches = face_recognition.compare_faces(
                        info["encodings"], current_encoding, tolerance=TOLERANCE
                    )
                    if any(matches):
                        matched = True
                        # 首次匹配：记录签到时间
                        if student_id not in signed_students:
                            sign_time = datetime.now().strftime("%H:%M:%S")
                            student_db[student_id]["attendance"] = "已签到"
                            student_db[student_id]["sign_time"] = sign_time
                            signed_students.add(student_id)
                            print(f"【签到成功】{info['name']}（学号：{student_id}）- {sign_time}")

                        # 绘制人脸框和姓名（绿色）
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, info["name"], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        break

                # 未匹配到学生库
                if not matched:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "未知人员", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 绘制考勤状态信息
        cv2.putText(frame, f"课堂考勤 | 已签到：{len(signed_students)}/{len(student_db)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"开始时间：{start_time}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        # 显示画面
        cv2.imshow("课堂人脸识别考勤系统", frame)

        # 键盘操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            # 按Q退出
            print("\n考勤结束！")
            break
        elif key == ord("s"):
            # 按S保存报表
            create_attendance_report(student_db, class_name)

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 最终保存报表
    create_attendance_report(student_db, class_name)
    print(f"最终考勤结果：已签到 {len(signed_students)} 人，未签到 {len(student_db) - len(signed_students)} 人")


# ===================== 主程序 =====================
if __name__ == "__main__":
    # 1. 加载学生人脸库
    student_database = load_student_database()

    # 2. 启动考勤系统
    if student_database:
        run_attendance_system(student_database, class_name="计算机2301班")
    else:
        print("请先在 dataset/students 目录下添加学生人脸照片！")