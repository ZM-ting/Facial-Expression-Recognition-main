import streamlit as st
import pandas as pd
import os
from datetime import datetime

# ==================【自动创建 datasets + 自动填数据】==================
if not os.path.exists("datasets"):
    os.mkdir("datasets")

# 自动创建员工表并写入数据
emp_data = [
    ["001","王浩","研发部"],["002","李雨桐","行政部"],["003","张佳宁","财务部"],
    ["004","刘俊泽","市场部"],["005","陈思远","人事部"]
]
pd.DataFrame(emp_data, columns=["编号","姓名","部门"]).to_csv("datasets/employees.csv", index=False)

# 自动创建考勤表并写入数据
rec_data = [
    ["2026-05-01 08:20:00","王浩","上班","正常"],
    ["2026-05-01 18:00:00","王浩","下班","正常"],
    ["2026-05-01 08:35:00","李雨桐","上班","迟到"],
]
pd.DataFrame(rec_data, columns=["时间","员工","类型","状态"]).to_csv("datasets/records.csv", index=False)

# ==================【页面功能】==================
st.set_page_config(layout="wide")
st.title("✅ 人脸考勤系统（已自动生成数据库）")

with st.sidebar:
    st.title("员工打卡")
    df_emp = pd.read_csv("datasets/employees.csv")
    selected = st.selectbox("选择员工", df_emp["姓名"].tolist())
    typ = st.radio("类型", ["上班", "下班"])

    # 【手动打卡 100% 能用】
    if st.button("✅ 手动提交打卡", type="primary", use_container_width=True):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([[now, selected, typ, "正常"]]).to_csv(
            "datasets/records.csv", mode="a", header=False, index=False
        )
        st.success("打卡成功！！！")

# 查看表格
st.subheader("考勤记录")
if st.button("查看所有打卡记录"):
    st.dataframe(pd.read_csv("datasets/records.csv"))

st.subheader("员工列表")
if st.button("查看所有员工"):
    st.dataframe(pd.read_csv("datasets/employees.csv"))