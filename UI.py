"""
===============================
⚠️ 已停用
此界面已替换为网页版：app.py
运行方式：streamlit run app.py
===============================
"""

# 全部注释，不运行
# 下面代码永久禁用
'''
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys

class OldUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("已停用")
        self.setFixedSize(300,200)
        label = QLabel("请使用 streamlit run app.py", self)
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OldUI()
    window.show()
    app.exec_()
'''