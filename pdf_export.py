from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib import colors
import pandas as pd


def export_attendance_pdf(df, save_path):
    """导出考勤报表为PDF"""
    # 创建PDF文档
    doc = SimpleDocTemplate(save_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # 标题
    title = Paragraph("员工考勤汇总报表", styles['Title'])

    # 数据表格
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    # 表格样式
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    # 构建文档内容
    elements = [title, table]
    doc.build(elements)
    return True

# 使用示例：
# df = pd.read_excel("考勤记录.xlsx")
# export_attendance_pdf(df, "考勤报表.pdf")