import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import pandas as pd
import time
from datetime import datetime
import altair as alt

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide"
)

# =========================================================
# PREMIUM FUTURISTIC CSS
# =========================================================

st.markdown("""
<style>

/* =====================================================
GLOBAL
===================================================== */

.stApp {

    background:
    radial-gradient(circle at top left, #172033 0%, #0F172A 40%, #020617 100%);

    color: white;
}

html, body, [class*="css"] {

    font-family: 'Inter', sans-serif;

    color: white;
}

/* =====================================================
CONTAINER
===================================================== */

.block-container {

    padding-top: 1.5rem;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* =====================================================
HEADINGS
===================================================== */

h1 {

    font-size: 3rem !important;

    font-weight: 800 !important;

    color: white;

    letter-spacing: -1px;
}

h2 {

    color: #E2E8F0;

    font-weight: 700 !important;
}

/* =====================================================
SIDEBAR
===================================================== */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        rgba(15,23,42,0.98),
        rgba(2,6,23,1)
    );

    border-right:
    1px solid rgba(255,255,255,0.08);

    backdrop-filter: blur(18px);
}

section[data-testid="stSidebar"] * {

    color: white !important;
}

/* =====================================================
METRIC CARDS
===================================================== */

[data-testid="metric-container"] {

    background:
    rgba(255,255,255,0.05);

    backdrop-filter:
    blur(18px);

    border:
    1px solid rgba(255,255,255,0.08);

    border-radius:
    24px;

    padding:
    22px;

    box-shadow:
    0 8px 32px rgba(0,0,0,0.45);

    transition:
    all 0.3s ease;
}

[data-testid="metric-container"]:hover {

    transform:
    translateY(-6px);

    border:
    1px solid #3B82F6;

    box-shadow:
    0 0 25px rgba(59,130,246,0.45);
}

/* =====================================================
DATAFRAME
===================================================== */

div[data-testid="stDataFrame"] {

    border-radius:
    20px;

    overflow:
    hidden;

    border:
    1px solid rgba(255,255,255,0.08);
}

/* =====================================================
UPLOAD
===================================================== */

[data-testid="stFileUploader"] {

    border:
    2px dashed rgba(255,255,255,0.15);

    border-radius:
    20px;

    padding:
    15px;

    background:
    rgba(255,255,255,0.03);
}

/* =====================================================
BUTTONS
===================================================== */

.stButton > button {

    width:
    100%;

    border-radius:
    14px;

    border:
    none;

    padding:
    0.7rem 1rem;

    background:
    linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    );

    color:
    white;

    font-weight:
    700;

    transition:
    all 0.3s ease;
}

.stButton > button:hover {

    transform:
    scale(1.03);

    box-shadow:
    0 0 20px rgba(59,130,246,0.5);
}

/* =====================================================
TABS
===================================================== */

.stTabs [data-baseweb="tab-list"] {

    gap:
    12px;
}

.stTabs [data-baseweb="tab"] {

    background:
    rgba(255,255,255,0.04);

    border-radius:
    14px;

    padding:
    12px 24px;

    border:
    1px solid rgba(255,255,255,0.06);
}

.stTabs [aria-selected="true"] {

    background:
    linear-gradient(
        135deg,
        #2563EB,
        #3B82F6
    ) !important;

    color:
    white !important;

    box-shadow:
    0 0 20px rgba(59,130,246,0.45);
}

/* =====================================================
ALERTS
===================================================== */

.stAlert {

    border-radius:
    18px;

    backdrop-filter:
    blur(12px);
}

/* =====================================================
IMAGES
===================================================== */

img {

    border-radius: 20px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
    0 0 30px rgba(59,130,246,0.15);
}

/* =====================================================
SCROLLBAR
===================================================== */

::-webkit-scrollbar {

    width: 10px;
}

::-webkit-scrollbar-track {

    background: #0F172A;
}

::-webkit-scrollbar-thumb {

    background: #334155;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {

    background: #3B82F6;
}

/* =====================================================
ANIMATION
===================================================== */

@keyframes fadeUp {

    from {

        opacity: 0;

        transform: translateY(20px);
    }

    to {

        opacity: 1;

        transform: translateY(0px);
    }
}

@keyframes pulseGlow {

    0% {

        box-shadow:
        0 0 10px rgba(59,130,246,0.2);
    }

    50% {

        box-shadow:
        0 0 25px rgba(59,130,246,0.45);
    }

    100% {

        box-shadow:
        0 0 10px rgba(59,130,246,0.2);
    }
}

[data-testid="metric-container"],
div[data-testid="stDataFrame"],
.stTabs,
.stAlert {

    animation:
    fadeUp 0.6s ease;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

from ultralytics import YOLO

with st.spinner("🚀 Initializing AI Traffic Intelligence..."):
    time.sleep(2)

model = YOLO("yolov8n.pt")
# =========================================================
# CLASS NAMES
# =========================================================

class_names = {
    0: 'auto',
    1: 'two_wheelers',
    2: 'bus',
    3: 'vehicle_truck',
    4: 'car',
    5: 'tractor',
    6: 'bicycle',
    7: 'tempo',
    8: 'ambulance'
}

# =========================================================
# BOX COLORS
# =========================================================

box_colors = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (128, 0, 255),
    7: (255, 128, 0),
    8: (0, 0, 255)
}

# =========================================================
# ELITE AI HEADER
# =========================================================

st.markdown("""

<div style="
padding:32px;
border-radius:30px;
background:rgba(255,255,255,0.05);
backdrop-filter:blur(18px);
border:1px solid rgba(255,255,255,0.08);
margin-bottom:20px;
animation:pulseGlow 3s infinite;
position:relative;
overflow:hidden;
">

<div style="
position:absolute;
top:-40px;
right:-40px;
width:180px;
height:180px;
background:rgba(59,130,246,0.18);
filter:blur(70px);
border-radius:50%;
"></div>

<div style="
display:flex;
justify-content:space-between;
align-items:center;
flex-wrap:wrap;
gap:20px;
">

<div>

<h1 style="
margin-bottom:5px;
font-size:52px;
font-weight:800;
color:white;
">

🚦 AI Smart Traffic Management

</h1>

<p style="
font-size:18px;
color:#94A3B8;
margin-top:10px;
">

Real-Time Vehicle Detection • AI Monitoring • Smart City Analytics

</p>

<div style="
margin-top:18px;
display:flex;
gap:12px;
flex-wrap:wrap;
">

<div style="
padding:8px 16px;
border-radius:999px;
background:rgba(34,197,94,0.15);
border:1px solid #22C55E;
color:#22C55E;
font-weight:600;
font-size:14px;
">

🟢 SYSTEM ACTIVE

</div>

<div style="
padding:8px 16px;
border-radius:999px;
background:rgba(59,130,246,0.15);
border:1px solid #3B82F6;
color:#3B82F6;
font-weight:600;
font-size:14px;
">

🤖 AI ENGINE RUNNING

</div>

<div style="
padding:8px 16px;
border-radius:999px;
background:rgba(245,158,11,0.15);
border:1px solid #F59E0B;
color:#F59E0B;
font-weight:600;
font-size:14px;
">

⚡ LIVE ANALYTICS

</div>

</div>

</div>

<div style="
text-align:right;
">

<div style="
font-size:14px;
color:#94A3B8;
margin-bottom:10px;
">

SMART CITY CONTROL CENTER

</div>

<div style="
width:90px;
height:90px;
border-radius:50%;
background:
linear-gradient(
135deg,
#2563EB,
#3B82F6
);
display:flex;
align-items:center;
justify-content:center;
font-size:40px;
box-shadow:
0 0 30px rgba(59,130,246,0.45);
animation:pulseGlow 2s infinite;
">

🚦

</div>

</div>

</div>

</div>

""", unsafe_allow_html=True)
# clock_col1, clock_col2 = st.columns([4,1])

# with clock_col2:
#     st.markdown("""
#     <div style="
#     margin-top:10px;
#     ">
#     """, unsafe_allow_html=True)

#     # LIVE CLOCK HERE

#     st.markdown("</div>", unsafe_allow_html=True)

current_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")

st.markdown(f"""

<div style="
display:flex;
justify-content:space-between;
align-items:center;
padding:16px 24px;
margin-top:12px;
margin-bottom:20px;
border-radius:20px;
background:rgba(255,255,255,0.04);
border:1px solid rgba(255,255,255,0.08);
backdrop-filter:blur(14px);
">

<div style="
color:#22C55E;
font-weight:600;
font-size:16px;
">

🟢 AI Traffic Intelligence Active

</div>

<div style="
color:#3B82F6;
font-weight:600;
font-size:16px;
">

🕒 {current_time}

</div>

<div style="
color:#F59E0B;
font-weight:600;
font-size:16px;
">

📡 Live Monitoring Enabled

</div>

</div>

""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# ⚙ Control Center")

    st.markdown("""

<div style="
display:flex;
justify-content:space-between;
align-items:center;
padding:16px 22px;
margin-top:15px;
margin-bottom:20px;
border-radius:20px;
background:rgba(255,255,255,0.04);
border:1px solid rgba(255,255,255,0.08);
backdrop-filter:blur(14px);
">

<div style="color:#22C55E;font-weight:600;">
🟢 AI Model Active
</div>

<div style="color:#3B82F6;font-weight:600;">
📡 Real-Time Monitoring Enabled
</div>

<div style="color:#F59E0B;font-weight:600;">
⚡ Smart Traffic Analytics Running
</div>

</div>

""", unsafe_allow_html=True)

    mode = st.selectbox(
        "Select Detection Mode",
        ["Image Detection", "Video Detection"]
    )

    camera_source = st.radio(
        "Select Video Source",
        ["Upload Video", "Webcam", "IP Camera"]
    )

    ip_url = None

    if camera_source == "IP Camera":

        ip_url = st.text_input(
            "Enter IP Camera URL",
            "http://192.168.0.100:8080/video"
        )

    st.divider()

    st.success("✅ YOLO Model Active")
    st.info("🚀 System Running")

# =========================================================
# KPI SECTION
# =========================================================

kpi_section = st.container()

# =========================================================
# IMAGE DETECTION
# =========================================================

if mode == "Image Detection":

    uploaded_image = st.file_uploader(
        "📤 Upload Traffic Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes, 1)

        frame = cv2.resize(frame, (1000, 550))

        results = model(frame)

        boxes = results[0].boxes

        counts = {name: 0 for name in class_names.values()}

        emergency_detected = False

        for box in boxes:

            cls = int(box.cls[0])

            class_name = class_names[cls]

            counts[class_name] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            color = box_colors[cls]

            if class_name == "ambulance":
                emergency_detected = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        total_vehicles = sum(counts.values())

        if total_vehicles < 10:
            traffic_status = "LOW"

        elif total_vehicles < 25:
            traffic_status = "MEDIUM"

        else:
            traffic_status = "HIGH"

        # =====================================================
        # KPI ROW
        # =====================================================

        with kpi_section:

            k1, k2, k3 = st.columns(3)

            with k1:
                st.metric("🚗 Vehicles", total_vehicles)

            with k2:
                st.metric("🚦 Traffic", traffic_status)

            with k3:
                st.metric(
                    "🚑 Emergency",
                    "ACTIVE" if emergency_detected else "NONE"
                )
        # =====================================================
        # AI STATUS PANEL
        # =====================================================

        st.markdown("""

        <div style="
        padding:18px;
        border-radius:22px;
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:20px;
        ">

        <h3 style="color:white;">
        🤖 AI System Diagnostics
        </h3>

        <ul style="color:#CBD5E1;line-height:2;">

        <li>YOLO Detection Engine: ACTIVE</li>
        <li>Vehicle Tracking: RUNNING</li>
        <li>Emergency Monitoring: ENABLED</li>
        <li>Smart Analytics: LIVE</li>
        <li>Real-Time Feed Processing: ACTIVE</li>

        </ul>

        </div>

        """, unsafe_allow_html=True)

        st.divider()
        st.divider()

        # =====================================================
        # MAIN GRID
        # =====================================================

        left_panel, right_panel = st.columns([7,3])

        with left_panel:

            st.markdown("## 🎥 Detection Output")

            st.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        with right_panel:

            st.markdown("## 📊 Analytics")

            df = pd.DataFrame(
                list(counts.items()),
                columns=["Vehicle Type", "Count"]
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            chart = alt.Chart(df).mark_bar(
    cornerRadiusTopLeft=10,
    cornerRadiusTopRight=10
).encode(
    x='Vehicle Type',
    y='Count',
    color='Vehicle Type'
)

            st.altair_chart(
                chart,
                use_container_width=True
            )

# =========================================================
# VIDEO DETECTION
# =========================================================

elif mode == "Video Detection":

    cap = None

    if camera_source == "Upload Video":

        uploaded_video = st.file_uploader(
            "📤 Upload Traffic Video",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_video:

            tfile = tempfile.NamedTemporaryFile(delete=False)

            tfile.write(uploaded_video.read())

            cap = cv2.VideoCapture(tfile.name)

    elif camera_source == "Webcam":

        cap = cv2.VideoCapture(0)

    elif camera_source == "IP Camera":

        if ip_url:
            cap = cv2.VideoCapture(ip_url)

    # =====================================================
    # VIDEO PROCESSING
    # =====================================================

    if cap is not None and cap.isOpened():

        video_container = st.container()

        bottom_container = st.container()

        counted_ids = set()

        total_counts = {
            name: 0 for name in class_names.values()
        }

        object_positions = {}

        accident_detected = False

        prev_time = time.time()

        frame_count = 0

        with video_container:

            left_panel, right_panel = st.columns([7,3])

            with left_panel:

                st.markdown("## 🎥 Live Traffic Feed")

                video_placeholder = st.empty()

            with right_panel:

                st.markdown("## 📈 Live Analytics")

                chart_placeholder = st.empty()

                st.markdown("## 🚨 System Alerts")

                alert_placeholder = st.empty()

        with bottom_container:

            tab1, tab2 = st.tabs([
                "Vehicle Data",
                "Traffic Insights"
            ])

            with tab1:
                table_placeholder = st.empty()

            with tab2:
                insights_placeholder = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (1000, 550))

            emergency_detected = False

            results = model.track(
                frame,
                persist=True,
                verbose=False
            )

            boxes = results[0].boxes

            current_time = time.time()

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy()

                classes = boxes.cls.cpu().numpy()

                xyxy = boxes.xyxy.cpu().numpy()

                for i, box in enumerate(xyxy):

                    x1, y1, x2, y2 = map(int, box)

                    obj_id = int(ids[i])

                    cls = int(classes[i])

                    class_name = class_names[cls]

                    if class_name == "ambulance":
                        emergency_detected = True

                    cx = int((x1 + x2) / 2)

                    cy = int((y1 + y2) / 2)

                    if obj_id not in counted_ids:

                        counted_ids.add(obj_id)

                        total_counts[class_name] += 1

                    speed = 0

                    if obj_id in object_positions:

                        prev_x, prev_y = object_positions[obj_id]

                        distance = np.sqrt(
                            (cx - prev_x)**2 +
                            (cy - prev_y)**2
                        )

                        time_diff = current_time - prev_time

                        speed = distance / (time_diff + 1e-5)

                        box_width = x2 - x1

                        box_height = y2 - y1

                        aspect_ratio = box_width / (box_height + 1e-5)

                        if speed > 150 and aspect_ratio > 2:
                            accident_detected = True

                    object_positions[obj_id] = (cx, cy)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_colors[cls],
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{class_name} | Speed:{speed:.1f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_colors[cls],
                        2
                    )
                    # =====================================================
                    # AI TRACKING HUD
                    # =====================================================

                    cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

                    cv2.line(frame, (cx, cy), (cx+20, cy), (0,255,255), 2)

                    cv2.putText(
                        frame,
                        f"ID:{obj_id}",
                        (cx+25, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,255),
                        2
                    )



            # =====================================================
            # FPS COUNTER
            # =====================================================

            fps = 1 / (time.time() - prev_time + 1e-5)

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            video_placeholder.image(
                frame,
                channels="BGR",
                
                use_container_width=True
            )
            prev_time = current_time

            frame_count += 1

            if frame_count % 10 == 0:

                df = pd.DataFrame(
                    list(total_counts.items()),
                    columns=["Vehicle Type", "Count"]
                )

                total_vehicle_number = sum(total_counts.values())

                if total_vehicle_number < 10:
                    traffic_status = "LOW"

                elif total_vehicle_number < 25:
                    traffic_status = "MEDIUM"

                else:
                    traffic_status = "HIGH"

                with kpi_section:

                    kp1, kp2, kp3, kp4 = st.columns(4)

                    with kp1:
                        st.metric(
                            "🚗 Vehicles",
                            total_vehicle_number
                        )

                    with kp2:
                        st.metric(
                            "🚦 Traffic",
                            traffic_status
                        )

                    with kp3:
                        st.metric(
                            "🚑 Emergency",
                            "ACTIVE" if emergency_detected else "NONE"
                        )

                    with kp4:
                        st.metric(
                            "⚠ Accident",
                            "DETECTED" if accident_detected else "SAFE"
                        )

                table_placeholder.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                chart = alt.Chart(df).mark_bar(
                    cornerRadiusTopLeft=10,
                    cornerRadiusTopRight=10
                ).encode(
                    x=alt.X('Vehicle Type', sort='-y'),
                    y='Count',
                    color='Vehicle Type'
                ).properties(
                    height=320
                )

                chart_placeholder.altair_chart(
                    chart,
                    use_container_width=True
                )

                if emergency_detected:

                    alert_placeholder.markdown("""

                    <div style="
                    padding:18px;
                    border-radius:18px;
                    background:rgba(239,68,68,0.15);
                    border:1px solid #EF4444;
                    color:white;
                    font-size:18px;
                    ">

                    🚑 Emergency Vehicle Detected

                    </div>

                    """, unsafe_allow_html=True)

                elif accident_detected:

                    alert_placeholder.markdown("""

                    <div style="
                    padding:18px;
                    border-radius:18px;
                    background:rgba(245,158,11,0.15);
                    border:1px solid #F59E0B;
                    color:white;
                    font-size:18px;
                    ">

                    ⚠ Possible Accident Detected

                    </div>

                    """, unsafe_allow_html=True)

                else:

                    alert_placeholder.markdown("""

                    <div style="
                    padding:18px;
                    border-radius:18px;
                    background:rgba(34,197,94,0.15);
                    border:1px solid #22C55E;
                    color:white;
                    font-size:18px;
                    ">

                    ✅ Traffic Running Normally

                    </div>

                    """, unsafe_allow_html=True)

                insights_placeholder.markdown(f"""
                ### 🚦 Traffic Insights

                - Total Vehicles: **{total_vehicle_number}**
                - Traffic Density: **{traffic_status}**
                - Emergency Vehicle: **{'YES' if emergency_detected else 'NO'}**
                - Accident Alert: **{'YES' if accident_detected else 'NO'}**
                """)

        cap.release()

        st.success("✅ Video Processing Completed")

    else:

        st.warning(
            "⚠ Please select a valid video source or connect camera."
        )

st.markdown("""

<hr style="
border:1px solid rgba(255,255,255,0.08);
">

<center>

<p style="
color:#94A3B8;
font-size:15px;
">

🚦 AI Smart Traffic Management System

<br>

Built with Streamlit • YOLO • OpenCV • AI Analytics

</p>

</center>

""", unsafe_allow_html=True)