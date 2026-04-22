import cv2
# 测试摄像头
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("摄像头正常！")
    while True:
        ret, frame = cap.read()
        cv2.imshow("测试摄像头", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("摄像头打开失败！试试把0改成1")