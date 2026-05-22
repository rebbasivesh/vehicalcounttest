import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import tempfile
import csv
import glob
from datetime import datetime
from ultralytics import YOLO

# Import local classes & utility functions
from vehicle_detector import VehicleDetector
from utils import draw_bbox, draw_fps, draw_dashboard, check_intersection, CLASS_COLORS

# Set page configuration first
st.set_page_config(
    page_title="AI Traffic Monitor & Vehicle Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look (Glassmorphic dark design)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0E1117;
}

/* Glassmorphic metric cards */
.metric-card {
    background: rgba(31, 38, 53, 0.45);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 18px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    margin-bottom: 12px;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 255, 102, 0.4);
    box-shadow: 0 8px 32px 0 rgba(0, 255, 102, 0.1);
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00FF66;
    margin-bottom: 4px;
    text-shadow: 0 0 10px rgba(0, 255, 102, 0.2);
}
.metric-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* Sidebar styling overrides */
[data-testid="stSidebar"] {
    background-color: #111520 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Styled buttons */
.stButton>button {
    background: linear-gradient(135deg, #00FF66 0%, #00B344 100%) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 255, 102, 0.15) !important;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 255, 102, 0.35) !important;
}

/* Stop button styling */
.stop-btn>div>button {
    background: linear-gradient(135deg, #FF3B30 0%, #C7000B 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(255, 59, 48, 0.25) !important;
}
.stop-btn>div>button:hover {
    box-shadow: 0 6px 20px rgba(255, 59, 48, 0.45) !important;
}

/* Main premium banner */
.app-banner {
    background: linear-gradient(135deg, #1C2333 0%, #0F131C 100%);
    padding: 30px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-left: 6px solid #00FF66;
    margin-bottom: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
}
.app-banner h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
}
.app-banner p {
    margin: 8px 0 0 0;
    color: #A0AEC0;
    font-size: 1.05rem;
}

/* Footer elements */
.app-footer {
    text-align: center;
    padding: 30px 10px;
    color: #4A5568;
    font-size: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 50px;
}
.app-footer a {
    color: #00FF66;
    text-decoration: none;
    font-weight: 600;
}
.app-footer a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def find_models():
    """
    Scans directories relative to app.py to locate YOLOv8 model files.
    """
    local_pt = glob.glob("*.pt")
    # Check common locations (parent folder s:\DATS or subfolders)
    parent_pt = glob.glob("../*.pt") + glob.glob("../../*.pt")
    
    all_pt = list(set(local_pt + [os.path.basename(p) for p in parent_pt]))
    
    # Standard YOLO models as absolute backups
    standards = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for s in standards:
        if s not in all_pt:
            # Let it download automatically if selected
            all_pt.append(s)
            
    # Sort models by size/completeness for cleaner look
    all_pt.sort()
    return all_pt

def map_class_name(cls_id, model_names):
    """
    Robustly maps YOLO detection class names to required vehicle categories.
    Works for standard COCO models and custom-trained vehicle models.
    """
    name = model_names.get(cls_id, "Unknown").lower()
    if "car" in name:
        return "Car"
    elif any(x in name for x in ["motorcycle", "bicycle", "bike", "cycle", "scooter", "person"]):
        return "Bike"
    elif "bus" in name:
        return "Bus"
    elif "truck" in name or "scv" in name or "lcv" in name:
        return "Truck"
    elif any(x in name for x in ["auto", "rickshaw", "tuk-tuk", "tuk"]):
        return "Auto"
    else:
        return "Others"

# Dynamic detection and tracking to support dynamic thresholds from the sidebar
def detect_and_track_custom(detector, frame, conf_threshold, iou_threshold, vehicle_classes):
    """
    Identical parser to vehicle_detector.py but parameters are dynamically adjusted.
    """
    results = detector.model.track(
        frame, 
        classes=vehicle_classes, 
        persist=True, 
        verbose=False, 
        tracker="botsort.yaml", 
        conf=conf_threshold,
        iou=iou_threshold
    )
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Dynamic mapping of the class name
                mapped_label = map_class_name(cls_id, detector.class_names)
                
                detections.append({
                    "box": (x1, y1, x2, y2),
                    "class_id": cls_id,
                    "label": mapped_label,
                    "conf": conf,
                    "track_id": track_id
                })
    return detections

# Initialize session state variables
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "process_complete" not in st.session_state:
    st.session_state.process_complete = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "final_counts" not in st.session_state:
    st.session_state.final_counts = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None

# Header Banner
st.markdown("""
<div class="app-banner">
    <h1>AI Traffic Monitor & Vehicle Counting Suite</h1>
    <p>A production-ready YOLOv8 computer vision dashboard for traffic analytics, flow optimization, and detailed audit reports.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("### 🛠️ Configuration Panel")

# Model Loader
available_models = find_models()
selected_model = st.sidebar.selectbox(
    "Choose YOLOv8 Model",
    options=available_models,
    index=available_models.index("yolov8m.pt") if "yolov8m.pt" in available_models else 0,
    help="If a model isn't local, YOLO will download it automatically."
)

# Sidebar sliders for Line Calibration
st.sidebar.markdown("### 📏 Line Calibrator (Entry/Exit)")
st.sidebar.caption("Adjust detection bounds horizontally across the frame width (1020x600 px)")

entry_line_y = st.sidebar.slider(
    "Entry Line Height (Line 1 - Yellow)",
    min_value=50,
    max_value=550,
    value=300,
    step=10,
    help="When a vehicle crosses this line, it enters the monitoring zone."
)

exit_line_y = st.sidebar.slider(
    "Exit Line Height (Line 2 - Magenta)",
    min_value=50,
    max_value=550,
    value=450,
    step=10,
    help="When a vehicle crosses this line, it is counted and classified."
)

# Threshold settings
st.sidebar.markdown("### 🎚️ Detection Parameters")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.15, 0.05)
iou_threshold = st.sidebar.slider("IOU Tracker Threshold", 0.10, 0.95, 0.50, 0.05)

# Performance tuning
st.sidebar.markdown("### ⚡ Performance Tuning")
frame_skipping = st.sidebar.slider(
    "Frame Skipping Factor",
    min_value=1,
    max_value=5,
    value=1,
    help="1 = process every frame. 2 = process alternate frames (2x faster), etc. Ideal for Streamlit Cloud hosting."
)

# Define lines based on sliders (Horizontal segments)
line1_pt1, line1_pt2 = (0, entry_line_y), (1020, entry_line_y)
line2_pt1, line2_pt2 = (0, exit_line_y), (1020, exit_line_y)

# ----------------- MAIN INTERFACE -----------------

# Dynamic model description
st.info(f"💡 Selected Model: **`{selected_model}`** | Frame Skipping: **{frame_skipping}x** | Confidence Threshold: **{conf_threshold}**")

# File Upload Panel
uploaded_file = st.file_uploader("📂 Choose a traffic video file to begin", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save the file to a temporary file safely
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Read video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
        
    ret, first_frame = cap.read()
    cap.release()
    
    if ret:
        first_frame_resized = cv2.resize(first_frame, (1020, 600))
        
        # Overlay lines for live visual calibration preview
        preview_frame = first_frame_resized.copy()
        cv2.line(preview_frame, line1_pt1, line1_pt2, (0, 255, 255), 3) # Yellow
        cv2.line(preview_frame, line2_pt1, line2_pt2, (255, 0, 255), 3) # Magenta
        
        cv2.putText(preview_frame, "ENTRY LINE 1 (YELLOW)", (20, entry_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(preview_frame, "EXIT LINE 2 (MAGENTA)", (20, exit_line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Display the calibration view
        if not st.session_state.is_processing and not st.session_state.process_complete:
            st.markdown("### 📐 Step 1: Calibration Preview")
            st.caption("Change the line positions in the sidebar to realign the zones. Ensure Line 1 (Entry) sits above Line 2 (Exit) in the traffic direction.")
            st.image(preview_frame, channels="BGR", use_container_width=True)
            
            # Action button
            if st.button("🚀 Start Vehicle Detection"):
                st.session_state.is_processing = True
                st.session_state.stop_requested = False
                st.session_state.process_complete = False
                st.session_state.report_data = None
                st.session_state.final_counts = None
                st.session_state.processed_video_path = None
                st.rerun()

    # ----------------- LIVE PROCESSING LOOP -----------------
    if st.session_state.is_processing:
        st.markdown("### 🎬 Step 2: Live Processing Stream")
        
        # Sidebar controls are disabled during run
        st.sidebar.warning("⚠️ Processing is active. Sidebar adjustments are locked.")
        
        # Processing Columns
        col_video, col_realtime = st.columns([7, 5])
        
        with col_video:
            video_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            # Custom styled stop button (uses the class from CSS)
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("🛑 Force Stop"):
                st.session_state.stop_requested = True
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_realtime:
            st.markdown("#### 📊 Real-Time Metrics")
            
            # Grid of cards
            m1, m2 = st.columns(2)
            with m1:
                fps_card = st.empty()
                active_card = st.empty()
            with m2:
                total_card = st.empty()
                time_card = st.empty()
                
            st.markdown("#### 📈 Dynamic Class Counts")
            plotly_placeholder = st.empty()
            
            st.markdown("#### 📜 Recent Detection Logs")
            log_table_placeholder = st.empty()
            
        # Re-initialize detector
        with st.spinner("Initializing YOLO model... (Downloading if first time)"):
            try:
                # Load detector
                detector = VehicleDetector(model_path=selected_model)
            except Exception as e:
                st.error(f"Failed to load YOLO model: {e}")
                st.session_state.is_processing = False
                st.stop()
                
        # Processing variables
        cap = cv2.VideoCapture(video_path)
        
        # Output video generation path
        out_path = os.path.join(tempfile.gettempdir(), 'streamlit_output_processed.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps_video, (1020, 600))
        
        # State tracking lists
        track_history = {}
        active_vehicles = {} # track_id -> entry_time
        counted_ids = set()
        vehicle_logs = []
        
        counts = {
            "Car": 0,
            "Bike": 0,
            "Bus": 0,
            "Truck": 0,
            "Auto": 0,
            "Others": 0
        }
        
        start_time = time.time()
        prev_time = start_time
        frame_idx = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                
                # Check for stop request
                if st.session_state.stop_requested:
                    st.warning("Detection paused by user request.")
                    break
                    
                # Frame skipping optimization
                if frame_idx % frame_skipping != 0:
                    continue
                    
                frame_resized = cv2.resize(frame, (1020, 600))
                
                # Run dynamic tracking
                detections = detect_and_track_custom(
                    detector, 
                    frame_resized, 
                    conf_threshold, 
                    iou_threshold, 
                    detector.vehicle_classes
                )
                
                color_line1 = (0, 255, 255) # Yellow
                color_line2 = (255, 0, 255) # Magenta
                
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                for det in detections:
                    box = det["box"]
                    label = det["label"]
                    conf = det["conf"]
                    track_id = det["track_id"]
                    
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((cx, cy))
                    
                    if len(track_history[track_id]) > 30:
                        track_history[track_id].pop(0)
                        
                    if len(track_history[track_id]) >= 2:
                        prev_pt = track_history[track_id][-2]
                        curr_pt = track_history[track_id][-1]
                        
                        # Crossing Line 1 (Entry)
                        if track_id not in active_vehicles and track_id not in counted_ids:
                            if check_intersection(prev_pt, curr_pt, line1_pt1, line1_pt2):
                                active_vehicles[track_id] = current_time_str
                                color_line1 = (0, 255, 0) # Flash green
                                
                        # Crossing Line 2 (Exit / Counting)
                        elif track_id in active_vehicles:
                            if check_intersection(prev_pt, curr_pt, line2_pt1, line2_pt2):
                                entry_time = active_vehicles.pop(track_id)
                                counted_ids.add(track_id)
                                
                                # Increment counts
                                if label in counts:
                                    counts[label] += 1
                                else:
                                    counts["Others"] += 1
                                    
                                # Calculate crossing duration
                                try:
                                    t1 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S.%f")
                                    t2 = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f")
                                    duration = (t2 - t1).total_seconds()
                                except:
                                    duration = 0.0
                                    
                                vehicle_logs.append({
                                    "Vehicle ID": track_id,
                                    "Class": label,
                                    "Entry Time": entry_time.split()[1], # Just the time block
                                    "Exit Time": current_time_str.split()[1],
                                    "Duration (s)": round(duration, 2)
                                })
                                
                                color_line2 = (0, 255, 0) # Flash green
                                
                    # Annotations
                    if track_id in active_vehicles or track_id in counted_ids:
                        bbox_color = (0, 255, 0) if track_id in active_vehicles else None
                        draw_bbox(frame_resized, box, label, conf, track_id=track_id, color=bbox_color)
                        
                        pts = track_history[track_id]
                        for i in range(1, len(pts)):
                            cv2.line(frame_resized, pts[i-1], pts[i], (255, 255, 0), 2)
                            
                # Draw boundaries
                cv2.line(frame_resized, line1_pt1, line1_pt2, color_line1, 3)
                cv2.line(frame_resized, line2_pt1, line2_pt2, color_line2, 3)
                cv2.putText(frame_resized, "ENTRY ZONE", (10, entry_line_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line1, 2)
                cv2.putText(frame_resized, "EXIT ZONE", (10, exit_line_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_line2, 2)
                
                # FPS calculation
                curr_time_fps = time.time()
                fps = 1.0 / (curr_time_fps - prev_time) if (curr_time_fps - prev_time) > 0 else 0.0
                prev_time = curr_time_fps
                
                draw_fps(frame_resized, fps)
                
                # Write frame to output
                out.write(frame_resized)
                
                # Update stream display
                video_placeholder.image(frame_resized, channels="BGR", use_container_width=True)
                
                # Update progress
                prog_pct = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(prog_pct)
                status_text.text(f"Processed Frame {frame_idx}/{total_frames} ({int(prog_pct*100)}%)")
                
                # Update live metrics cards
                active_count = len(active_vehicles)
                total_count = len(counted_ids)
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                
                fps_card.markdown(f'<div class="metric-card"><div class="metric-value">{fps:.1f}</div><div class="metric-label">Live FPS</div></div>', unsafe_allow_html=True)
                active_card.markdown(f'<div class="metric-card"><div class="metric-value">{active_count}</div><div class="metric-label">Active Track</div></div>', unsafe_allow_html=True)
                total_card.markdown(f'<div class="metric-card"><div class="metric-value">{total_count}</div><div class="metric-label">Total Counted</div></div>', unsafe_allow_html=True)
                time_card.markdown(f'<div class="metric-card"><div class="metric-value">{minutes:02d}:{seconds:02d}</div><div class="metric-label">Elapsed Time</div></div>', unsafe_allow_html=True)
                
                # Live Plotly Graph
                counts_df = pd.DataFrame({
                    "Class": list(counts.keys()),
                    "Count": list(counts.values())
                })
                fig = px.bar(
                    counts_df, 
                    x="Count", 
                    y="Class", 
                    orientation='h', 
                    color="Class",
                    color_discrete_map={
                        "Car": "#2596be", "Bike": "#ffa500", "Bus": "#00ff00",
                        "Truck": "#ff00ff", "Auto": "#f7d038", "Others": "#808080"
                    },
                    height=240
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#888888')),
                    yaxis=dict(tickfont=dict(color='#888888'), title='')
                )
                plotly_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Recent detections log list
                if len(vehicle_logs) > 0:
                    recent_df = pd.DataFrame(vehicle_logs).tail(5).iloc[::-1] # Show last 5 logs reversed
                    log_table_placeholder.dataframe(recent_df, use_container_width=True, hide_index=True)
                else:
                    log_table_placeholder.info("Awaiting line crossing events...")
                    
        except Exception as e:
            st.error(f"An error occurred during tracking: {e}")
            
        finally:
            cap.release()
            out.release()
            
            # Save results into session state
            st.session_state.is_processing = False
            st.session_state.process_complete = True
            st.session_state.report_data = vehicle_logs
            st.session_state.final_counts = counts
            st.session_state.processed_video_path = out_path
            
            # Auto-cleanup uploaded raw temp file
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass
                    
            st.rerun()

    # ----------------- POST-PROCESSING ANALYTICS DASHBOARD -----------------
    if st.session_state.process_complete and st.session_state.report_data is not None:
        st.success("🎉 Video processing complete!")
        
        # Retrieve logs and counts
        logs = st.session_state.report_data
        counts = st.session_state.final_counts
        processed_video_path = st.session_state.processed_video_path
        
        # Summary Analytics
        total_vehicles = sum(counts.values())
        
        st.markdown("### 📊 Step 3: Complete Traffic Analytics Dashboard")
        
        # Big metric summaries
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_vehicles}</div><div class="metric-label">Total Vehicles Audited</div></div>', unsafe_allow_html=True)
        with sc2:
            peak_class = max(counts, key=counts.get) if total_vehicles > 0 else "N/A"
            st.markdown(f'<div class="metric-card"><div class="metric-value">{peak_class}</div><div class="metric-label">Dominant Class</div></div>', unsafe_allow_html=True)
        with sc3:
            avg_duration = round(np.mean([x["Duration (s)"] for x in logs]), 2) if len(logs) > 0 else 0.0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_duration}s</div><div class="metric-label">Average Transit Duration</div></div>', unsafe_allow_html=True)
            
        # Graphical Analytics
        ac1, ac2 = st.columns(2)
        
        with ac1:
            st.markdown("#### Vehicle Class Distribution")
            pie_df = pd.DataFrame({"Class": list(counts.keys()), "Count": list(counts.values())})
            pie_df = pie_df[pie_df["Count"] > 0]
            
            if len(pie_df) > 0:
                fig_pie = px.pie(
                    pie_df, 
                    values="Count", 
                    names="Class", 
                    color="Class",
                    color_discrete_map={
                        "Car": "#2596be", "Bike": "#ffa500", "Bus": "#00ff00",
                        "Truck": "#ff00ff", "Auto": "#f7d038", "Others": "#808080"
                    },
                    hole=0.4
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color='#ffffff'))
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No vehicles were detected to display charts.")
                
        with ac2:
            st.markdown("#### Traffic Flow Over Time (Temporal Volume)")
            if len(logs) > 0:
                log_df = pd.DataFrame(logs)
                # Convert string exit times to relative seconds/minutes to show distribution
                log_df["Exit Time Sec"] = log_df["Exit Time"].apply(lambda x: sum(int(t) * 60**i for i, t in enumerate(reversed(x.split(':')))))
                # Group into 5-second intervals
                log_df["Interval"] = (log_df["Exit Time Sec"] // 5) * 5
                temporal_df = log_df.groupby("Interval").size().reset_index(name="Volume")
                
                fig_line = px.line(
                    temporal_df, 
                    x="Interval", 
                    y="Volume", 
                    markers=True,
                    labels={"Interval": "Time Interval (relative seconds)", "Volume": "Vehicles Per 5s"}
                )
                fig_line.update_traces(line_color="#00FF66", marker=dict(size=8, color="#ffffff"))
                fig_line.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#888888')),
                    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#888888'))
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Awaiting tracking timeline logs.")
                
        # Detailed Logs & CSV Download
        st.markdown("---")
        st.markdown("#### 📋 Comprehensive Traffic Audit Log")
        
        if len(logs) > 0:
            full_df = pd.DataFrame(logs)
            st.dataframe(full_df, use_container_width=True, hide_index=True)
            
            # Prepare CSV report download in memory
            csv_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w')
            writer = csv.writer(csv_buffer)
            writer.writerow(["Vehicle ID", "Class", "Entry Time", "Exit Time", "Transit Duration (sec)"])
            for item in logs:
                writer.writerow([item["Vehicle ID"], item["Class"], item["Entry Time"], item["Exit Time"], item["Duration (s)"]])
                
            writer.writerow([])
            writer.writerow(["--- SUMMARY STATS ---"])
            for k, v in counts.items():
                writer.writerow([k, f"Total counted: {v}"])
            writer.writerow(["TOTAL AUDITED", total_vehicles])
            csv_buffer.close()
            
            # Read back buffer for download button
            with open(csv_buffer.name, 'r') as f:
                csv_str = f.read()
                
            st.download_button(
                label="📥 Download Detailed CSV Audit Report",
                data=csv_str,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Attempt to clean up csv buffer
            try:
                os.remove(csv_buffer.name)
            except:
                pass
        else:
            st.info("No vehicles logged.")
            
        # Download Output Video Panel
        st.markdown("---")
        st.markdown("#### 🎥 Processed Video Export")
        if processed_video_path and os.path.exists(processed_video_path):
            with open(processed_video_path, 'rb') as f:
                video_bytes = f.read()
                
            st.download_button(
                label="📥 Download Processed Annotated Video",
                data=video_bytes,
                file_name="traffic_detection_output.mp4",
                mime="video/mp4"
            )
            
        # Button to allow another run
        st.markdown("---")
        if st.button("🔄 Analyze Another Video"):
            st.session_state.is_processing = False
            st.session_state.process_complete = False
            st.session_state.report_data = None
            st.session_state.final_counts = None
            st.session_state.stop_requested = False
            
            # Cleanup processed output temp file
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                try:
                    os.remove(st.session_state.processed_video_path)
                except:
                    pass
            st.session_state.processed_video_path = None
            st.rerun()

else:
    # Home default info panel when no video is loaded
    st.markdown("---")
    st.markdown("### 🚦 Quick-Start Instructions")
    
    col_inst1, col_inst2, col_inst3 = st.columns(3)
    with col_inst1:
        st.markdown("""
        **1. Configure Model & Lines**
        - Select your desired **YOLOv8** model from the sidebar.
        - Tweak the entry (Line 1) and exit (Line 2) height boundaries to fit the traffic camera's perspective.
        """)
    with col_inst2:
        st.markdown("""
        **2. Upload a Traffic Feed**
        - Drag-and-drop or upload your traffic footage above.
        - Supported video files: `.mp4`, `.avi`, `.mov`, `.mkv`.
        - The app will render a real-time calibration preview immediately.
        """)
    with col_inst3:
        st.markdown("""
        **3. Run & Export Audits**
        - Hit **Start Vehicle Detection**.
        - Watch the live stream track and log vehicles in real-time.
        - Instantly download dynamic Plotly summaries and complete `.csv` spreadsheet logs.
        """)

# Clean, professional footer
st.markdown("""
<div class="app-footer">
    AI Traffic Monitor &copy; 2026. Powered by <a href="https://ultralytics.com/" target="_blank">Ultralytics YOLOv8</a> & <a href="https://streamlit.io/" target="_blank">Streamlit</a>.<br>
    Ready for deployment to <b>GitHub</b> and <b>Streamlit Community Cloud</b>.
</div>
""", unsafe_allow_html=True)
