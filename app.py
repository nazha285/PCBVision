import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import openpyxl
import io

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime

# ==========================
# YOLO CLASS COLORS
# ==========================
YOLO_COLORS = {
    "open_circuit":    "#FF3838",
    "short":           "#FF9D97",
    "missing_hole":    "#FF701F",
    "mouse_bite":      "#FFB21D",
    "spur":            "#CFD231",
    "spurious_copper": "#48F90A",
}

# ==========================
# LOAD MODEL
# ==========================
model = YOLO("runs/detect/train-6/weights/best.pt")

# ==========================
# DATABASE SETUP
# ==========================
conn = sqlite3.connect("inspection_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS inspections (
    id      TEXT,
    date    TEXT,
    defects INTEGER,
    quality INTEGER,
    status  TEXT
)
""")
conn.commit()

# ==========================
# PAGE SETTINGS
# ==========================
st.set_page_config(
    page_title="PCB Vision — AI Defect Inspector",
    page_icon="🔬",
    layout="wide"
)

# ==========================
# CUSTOM CSS
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}
.stApp { background-color: #0a0e1a; }
#MainMenu, footer, header { visibility: hidden; }

.top-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 60%, #0a192f 100%);
    border-bottom: 1px solid #1e3a5f;
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -1rem 2rem -1rem;
}
.banner-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.18em; color: #38bdf8;
    text-transform: uppercase; margin-bottom: 0.6rem;
}
.banner-title {
    font-size: 2.4rem; font-weight: 700;
    color: #f0f6ff; letter-spacing: -0.02em;
    line-height: 1.1; margin-bottom: 0.4rem;
}
.banner-title span { color: #38bdf8; }
.banner-sub {
    font-size: 0.95rem; color: #7fa8c9;
    font-weight: 400; margin-bottom: 1.2rem;
}
.banner-badges { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; font-weight: 600;
    padding: 0.25rem 0.7rem; border-radius: 999px;
    border: 1px solid; letter-spacing: 0.05em;
}
.badge-blue   { color: #38bdf8; border-color: #1e4a6e; background: #0c2d45; }
.badge-green  { color: #34d399; border-color: #1a4a38; background: #0c2d22; }
.badge-purple { color: #a78bfa; border-color: #3b2f6e; background: #1e1545; }

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.15em; color: #38bdf8;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.stat-row { display: flex; gap: 1rem; margin-bottom: 1.25rem; }
.stat-card {
    flex: 1; background: #0a1526;
    border: 1px solid #1a2d45; border-radius: 10px;
    padding: 1.1rem 1.25rem; text-align: center;
}
.stat-label {
    font-size: 0.72rem; color: #64748b; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #38bdf8; line-height: 1;
}
.stat-value.red   { color: #f87171; }
.stat-value.green { color: #34d399; }

.status-fail {
    background: linear-gradient(135deg, #2d0f0f, #1a0808);
    border: 1px solid #7f1d1d; border-left: 4px solid #ef4444;
    border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.status-pass {
    background: linear-gradient(135deg, #0f2d1a, #081a10);
    border: 1px solid #14532d; border-left: 4px solid #22c55e;
    border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}
.status-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem; font-weight: 700;
    letter-spacing: 0.05em; margin-bottom: 0.3rem;
}
.status-title.red   { color: #f87171; }
.status-title.green { color: #34d399; }
.status-desc { font-size: 0.85rem; color: #94a3b8; line-height: 1.6; }

.rec-list { display: flex; flex-direction: column; gap: 0.5rem; margin-top: 1rem; }
.rec-item {
    display: flex; align-items: center; gap: 0.75rem;
    background: #0a1526; border: 1px solid #1e3a5f;
    border-radius: 8px; padding: 0.65rem 1rem;
    font-size: 0.875rem; color: #cbd5e1;
}
.rec-icon { font-size: 1rem; }

[data-testid="stFileUploader"] {
    background: #0f1929 !important;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }
[data-testid="stDownloadButton"] button {
    background: #0f2d45 !important; color: #38bdf8 !important;
    border: 1px solid #1e4a6e !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid #1a2d45 !important;
    border-radius: 10px !important; overflow: hidden !important;
}
[data-testid="stProgress"] > div > div {
    background-color: #0ea5e9 !important; border-radius: 4px !important;
}
.img-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.12em; color: #38bdf8;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.footer {
    background: #080d18; border-top: 1px solid #1a2d45;
    padding: 2rem; margin: 3rem -1rem -1rem -1rem;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 1rem;
}
.footer-left { font-size: 0.85rem; color: #475569; }
.footer-right { display: flex; gap: 1rem; }
.footer-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #334155; letter-spacing: 0.08em;
}
[data-testid="stMetric"] {
    background: #0a1526; border: 1px solid #1a2d45;
    border-radius: 10px; padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.markdown("""
<div style="text-align:center;padding:1rem 0 0.5rem">
    <div style="font-family:JetBrains Mono,monospace;font-size:1.3rem;font-weight:700;color:#38bdf8">PCBVision</div>
    <div style="font-size:0.7rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">Industrial AOI System</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**👩‍💻 Developed by**")
st.sidebar.markdown("""
<div style="font-size:0.8rem;color:#94a3b8">Nazha Al Rajab<br>AI + Computer Vision</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Model Info**")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.05, max_value=0.90,
    value=0.10, step=0.05,
    help="Lower = more detections (may include false positives). Higher = fewer, more certain detections."
)

st.sidebar.markdown(f"""
<div style="font-size:0.8rem;color:#94a3b8;line-height:1.9">
Model: YOLOv8<br>
Classes: 6<br>
Conf Threshold: {conf_threshold:.2f}<br>
IoU Threshold: 0.45<br>
Trained on: DeepPCB
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**🔵 Defect Classes**")

for class_name, hex_color in YOLO_COLORS.items():
    st.sidebar.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;font-size:0.8rem;'
        f'color:#94a3b8;margin-bottom:6px">'
        f'<div style="width:13px;height:13px;border-radius:50%;'
        f'background:{hex_color};flex-shrink:0"></div>'
        f'{class_name.replace("_", " ").title()}</div>',
        unsafe_allow_html=True
    )

# ==========================
# TOP BANNER
# ==========================
st.markdown("""
<div class="top-banner">
    <div class="banner-eyebrow">Industrial AOI System</div>
    <div class="banner-title">PCB<span>Vision</span></div>
    <div class="banner-sub">AI-Powered Printed Circuit Board Defect Detection</div>
    <div class="banner-badges">
        <span class="badge badge-blue">YOLOv8</span>
        <span class="badge badge-green">Real-time Detection</span>
        <span class="badge badge-purple">DeepPCB Dataset</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# IMAGE UPLOAD
# ==========================
st.markdown('<div class="section-label">Step 1 — Input</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drop a PCB image here or click to browse",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="img-label">Original Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    if st.button("⚡  Run Defect Detection"):

        with st.spinner("Running inference..."):
            results = model.predict(image, conf=conf_threshold, iou=0.45)

        result_image   = results[0].plot()
        detected_image = Image.fromarray(result_image)
        detected_image.save("detected_result.jpg")

        inspection_id   = "PCB-" + datetime.now().strftime("%Y%m%d%H%M%S")
        inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with col2:
            st.markdown('<div class="img-label">Detection Result</div>', unsafe_allow_html=True)
            st.image(result_image, use_container_width=True)

        boxes       = results[0].boxes
        defects     = []
        confidences = []
        locations   = []

        for box in boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            defects.append(model.names[cls])
            confidences.append(round(conf * 100, 2))
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = round((x1 + x2) / 2, 1)
            cy = round((y1 + y2) / 2, 1)
            locations.append(f"({cx}, {cy})")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── FAIL ──
        if len(defects) > 0:

            quality_score = max(0, 100 - len(defects) * 10)

            cursor.execute(
                "INSERT INTO inspections VALUES (?, ?, ?, ?, ?)",
                (inspection_id, inspection_time, len(defects), quality_score, "FAIL")
            )
            conn.commit()

            st.markdown('<div class="section-label">Analysis</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-label">Total Defects</div>
                    <div class="stat-value red">{len(defects)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Highest Confidence</div>
                    <div class="stat-value">{max(confidences):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg Confidence</div>
                    <div class="stat-value">{np.mean(confidences):.1f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Quality Score</div>
                    <div class="stat-value {'green' if quality_score >= 80 else 'red'}">{quality_score}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            id_col, time_col = st.columns(2)
            with id_col:
                st.metric("🆔 Inspection ID", inspection_id)
            with time_col:
                st.metric("📅 Inspection Time", inspection_time)

            with open("detected_result.jpg", "rb") as img_file:
                st.download_button(
                    label="🖼️  Download Detected Image",
                    data=img_file,
                    file_name="detected_result.jpg",
                    mime="image/jpeg"
                )

            st.markdown("""
            <div class="status-fail">
                <div class="status-title red">🔴 PCB STATUS: FAIL</div>
                <div class="status-desc">Defects were detected on this board. Review the findings below before making a disposition decision.</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Confidence Levels ──
            st.markdown('<div class="section-label">Confidence Levels</div>', unsafe_allow_html=True)
            for defect, conf in zip(defects, confidences):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;'
                        f'color:#f59e0b;margin-bottom:4px">{defect}</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(conf / 100)
                with col_b:
                    st.markdown(
                        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.85rem;'
                        f'color:#94a3b8;padding-top:4px;text-align:right">{conf:.1f}%</div>',
                        unsafe_allow_html=True
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Detection Table ──
            st.markdown('<div class="section-label">Detection Log</div>', unsafe_allow_html=True)
            df = pd.DataFrame({
                "Defect Type":    defects,
                "Confidence (%)": confidences,
                "Location (x,y)": locations
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            excel_name = "inspection_report.xlsx"
            df.to_excel(excel_name, index=False)
            with open(excel_name, "rb") as excel_file:
                st.download_button(
                    label="📊  Download Excel Report",
                    data=excel_file,
                    file_name=excel_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # ── Defect Breakdown ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Defect Breakdown</div>', unsafe_allow_html=True)

            summary_df = pd.Series(defects).value_counts().reset_index()
            summary_df.columns = ["Defect Type", "Count"]

            col_s1, col_s2 = st.columns([1, 1], gap="large")
            with col_s1:
                st.markdown(
                    f'<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem">'
                    f'{len(summary_df)} distinct defect type(s)</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            with col_s2:
                if len(summary_df) >= 2:
                    palette = ["#38bdf8", "#f59e0b", "#f87171", "#34d399", "#a78bfa", "#fb923c"]
                    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#0a1526")
                    ax.set_facecolor("#0a1526")
                    wedges, texts, autotexts = ax.pie(
                        summary_df["Count"],
                        labels=summary_df["Defect Type"],
                        autopct="%1.1f%%",
                        colors=palette[:len(summary_df)],
                        startangle=90,
                        wedgeprops=dict(linewidth=2, edgecolor="#0a0e1a")
                    )
                    for t in texts:
                        t.set_color("#94a3b8"); t.set_fontsize(9)
                    for at in autotexts:
                        at.set_color("#e2e8f0"); at.set_fontsize(8); at.set_fontweight("bold")
                    st.pyplot(fig)

            if len(summary_df) >= 2:
                palette = ["#38bdf8", "#f59e0b", "#f87171", "#34d399", "#a78bfa", "#fb923c"]
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-label">📊 Defect Frequency</div>', unsafe_allow_html=True)
                fig_bar, ax_bar = plt.subplots(figsize=(6, 3), facecolor="#0a1526")
                ax_bar.set_facecolor("#0a1526")
                bars = ax_bar.bar(
                    summary_df["Defect Type"], summary_df["Count"],
                    color=palette[:len(summary_df)], edgecolor="#0a0e1a", linewidth=1.5
                )
                ax_bar.set_xlabel("Defect Type", color="#64748b", fontsize=9)
                ax_bar.set_ylabel("Count", color="#64748b", fontsize=9)
                ax_bar.tick_params(colors="#94a3b8", labelsize=8)
                ax_bar.spines[:].set_color("#1a2d45")
                for bar in bars:
                    ax_bar.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        str(int(bar.get_height())),
                        ha="center", va="bottom",
                        color="#e2e8f0", fontsize=9, fontweight="bold"
                    )
                st.pyplot(fig_bar)

            # ── Quality Indicator ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">⭐ Quality Indicator</div>', unsafe_allow_html=True)
            st.progress(quality_score / 100)
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.85rem;'
                f'color:#94a3b8;margin-top:4px">Quality Score = {quality_score}%</div>',
                unsafe_allow_html=True
            )

            # ── PDF Export ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)

            orig_buf = io.BytesIO()
            image.save(orig_buf, format="JPEG")
            orig_buf.seek(0)
            with open("_orig_temp.jpg", "wb") as f:
                f.write(orig_buf.read())
            detected_image.save("_det_temp.jpg")

            doc    = SimpleDocTemplate("inspection_report.pdf", rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            story  = []

            story.append(Paragraph("🔬 AI PCB Inspection Report", styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"<b>Inspection ID:</b> {inspection_id}", styles['Normal']))
            story.append(Paragraph(f"<b>Date &amp; Time:</b> {inspection_time}", styles['Normal']))
            story.append(Paragraph(f"<b>Total Defects:</b> {len(defects)}", styles['Normal']))
            story.append(Paragraph(f"<b>Quality Score:</b> {quality_score}%", styles['Normal']))
            story.append(Paragraph("<b>Final Assessment:</b> FAIL", styles['Normal']))
            story.append(Spacer(1, 16))
            story.append(Paragraph("Original Image", styles['Heading2']))
            story.append(RLImage("_orig_temp.jpg", width=3*inch, height=2.2*inch))
            story.append(Spacer(1, 8))
            story.append(Paragraph("Detection Result", styles['Heading2']))
            story.append(RLImage("_det_temp.jpg", width=3*inch, height=2.2*inch))
            story.append(Spacer(1, 16))
            story.append(Paragraph("Defect Log", styles['Heading2']))
            story.append(Spacer(1, 6))

            data = [["#", "Defect Type", "Confidence (%)", "Location (x,y)"]]
            for i, (d, c, l) in enumerate(zip(defects, confidences, locations), 1):
                data.append([str(i), d, str(c), l])

            table = Table(data, colWidths=[0.4*inch, 2*inch, 1.4*inch, 1.4*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0d3a5c")),
                ('TEXTCOLOR',  (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.HexColor("#dce8f0")]),
                ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor("#aac4d8")),
                ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE',   (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(table)
            doc.build(story)

            with open("inspection_report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📄  Download Inspection Report (PDF)",
                    data=pdf_file,
                    file_name="inspection_report.pdf",
                    mime="application/pdf"
                )

            # ── Disposition ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Disposition</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="rec-list">
                <div class="rec-item"><span class="rec-icon">❌</span> Reject PCB — board does not meet quality standards</div>
                <div class="rec-item"><span class="rec-icon">🔧</span> Send to rework station for repair</div>
                <div class="rec-item"><span class="rec-icon">🔍</span> Perform secondary manual inspection</div>
                <div class="rec-item"><span class="rec-icon">📋</span> Log findings and generate quality report</div>
            </div>
            """, unsafe_allow_html=True)

        # ── PASS ──
        else:
            cursor.execute(
                "INSERT INTO inspections VALUES (?, ?, ?, ?, ?)",
                (inspection_id, inspection_time, 0, 100, "PASS")
            )
            conn.commit()

            st.markdown("""
            <div class="status-pass">
                <div class="status-title green">🟢 PCB STATUS: PASS</div>
                <div class="status-desc">No defects detected above the confidence threshold. This board is cleared for production.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-label">Total Defects</div>
                    <div class="stat-value green">0</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Quality Score</div>
                    <div class="stat-value green">100%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            pass_id_col, pass_time_col = st.columns(2)
            with pass_id_col:
                st.metric("🆔 Inspection ID", inspection_id)
            with pass_time_col:
                st.metric("📅 Inspection Time", inspection_time)

            st.markdown("""
            <div class="rec-list">
                <div class="rec-item"><span class="rec-icon">✅</span> PCB Accepted — board meets all quality standards</div>
                <div class="rec-item"><span class="rec-icon">✅</span> Ready for production line</div>
                <div class="rec-item"><span class="rec-icon">✅</span> Quality Approved — no rework required</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================
# INSPECTION HISTORY
# ==========================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">📜 Inspection History</div>', unsafe_allow_html=True)

history = pd.read_sql_query("SELECT * FROM inspections ORDER BY date DESC", conn)

if len(history) > 0:
    st.dataframe(history, use_container_width=True, hide_index=True)

    history_excel = io.BytesIO()
    history.to_excel(history_excel, index=False)
    history_excel.seek(0)
    st.download_button(
        label="📥  Download Full History (Excel)",
        data=history_excel,
        file_name="inspection_history.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️  Clear Inspection History"):
        cursor.execute("DELETE FROM inspections")
        conn.commit()
        st.success("History cleared. Refresh the page to see the updated table.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🔍 Search Inspection History</div>', unsafe_allow_html=True)
    search_id = st.text_input("Enter Inspection ID to search", placeholder="e.g. PCB-20260612111530")
    if search_id:
        filtered = history[history["id"].astype(str).str.contains(search_id, case=False)]
        if len(filtered) > 0:
            st.dataframe(filtered, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div style="color:#f87171;font-size:0.85rem;padding:0.5rem">No matching inspection found.</div>',
                unsafe_allow_html=True
            )
else:
    st.markdown(
        '<div style="color:#475569;font-size:0.85rem;padding:1rem">No inspections recorded yet. Run a detection to start building history.</div>',
        unsafe_allow_html=True
    )

# ==========================
# PRODUCTION STATISTICS
# ==========================
if len(history) > 0:

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">🏭 Production Statistics</div>', unsafe_allow_html=True)

    passed_boards   = len(history[history["status"] == "PASS"])
    failed_boards   = len(history[history["status"] == "FAIL"])
    average_quality = round(history["quality"].mean(), 2)

    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1: st.metric("🔬 Total Inspected", len(history))
    with ps2: st.metric("🟢 Passed Boards", passed_boards)
    with ps3: st.metric("🔴 Failed Boards", failed_boards)
    with ps4: st.metric("⭐ Avg Quality", f"{average_quality}%")

    # ── Quality Trend (replaces pie chart) ──
    x_vals = list(range(1, len(history) + 1))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📈 Quality Trend</div>', unsafe_allow_html=True)
    fig_qt, ax_qt = plt.subplots(figsize=(8, 3), facecolor="#0a1526")
    ax_qt.set_facecolor("#0a1526")
    ax_qt.plot(
        x_vals, history["quality"].values[::-1],
        color="#38bdf8", linewidth=2, marker="o",
        markersize=4, markerfacecolor="#38bdf8"
    )
    ax_qt.set_xlabel("Inspection #", color="#64748b", fontsize=8)
    ax_qt.set_ylabel("Quality Score", color="#64748b", fontsize=8)
    ax_qt.tick_params(colors="#94a3b8", labelsize=7)
    ax_qt.spines[:].set_color("#1a2d45")
    ax_qt.set_ylim(0, 110)
    ax_qt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    st.pyplot(fig_qt)

    # ── Defect Trend ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">📉 Defect Trend</div>', unsafe_allow_html=True)
    fig_dt, ax_dt = plt.subplots(figsize=(8, 3), facecolor="#0a1526")
    ax_dt.set_facecolor("#0a1526")
    ax_dt.bar(
        x_vals, history["defects"].values[::-1],
        color="#f87171", edgecolor="#0a0e1a", linewidth=1
    )
    ax_dt.set_xlabel("Inspection #", color="#64748b", fontsize=8)
    ax_dt.set_ylabel("Defects Found", color="#64748b", fontsize=8)
    ax_dt.tick_params(colors="#94a3b8", labelsize=7)
    ax_dt.spines[:].set_color("#1a2d45")
    ax_dt.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    st.pyplot(fig_dt)

# ==========================
# FOOTER
# ==========================
st.markdown("""
<div class="footer">
    <div class="footer-left">Developed by <strong>Nazha Al Rajab</strong> &nbsp;·&nbsp; AI + Computer Vision + PCB Inspection</div>
    <div class="footer-right">
        <span class="footer-tag">YOLOv8</span>
        <span class="footer-tag">Streamlit</span>
        <span class="footer-tag">DeepPCB</span>
    </div>
</div>
""", unsafe_allow_html=True)