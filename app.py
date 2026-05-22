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
import uuid
import sys
from datetime import datetime

# ----------------- TRACKER DEBBUGGING FALLBACK SYSTEM -----------------
# Dynamically register lapx as lap in sys.modules to prevent the classic
# "No module named 'lap'" error in YOLOv8's internal Hungarian matcher.
try:
    import lap
except ImportError:
    try:
        import lapx as lap
        sys.modules['lap'] = lap  # Inject lapx into standard lap namespace
        print("[DEBUG] Successfully injected lapx as lap wrapper in app.py.", file=sys.stderr)
    except ImportError:
        print("[WARNING] Neither 'lap' nor 'lapx' is installed. YOLO tracking might fail.", file=sys.stderr)

# Set page configuration FIRST before any other streamlit call
st.set_page_config(
    page_title="AI Traffic Monitor & Vehicle Counting Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium black/white/neon-green minimalist interface
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #0B0E14;
    color: #FFFFFF;
}

/* Glassmorphic metric cards */
.metric-card {
    background: rgba(30, 36, 50, 0.4);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    transition: all 0.3s ease;
    margin-bottom: 12px;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 255, 102, 0.3);
    box-shadow: 0 8px 32px 0 rgba(0, 255, 102, 0.08);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #00FF66;
    margin-bottom: 4px;
    text-shadow: 0 0 10px rgba(0, 255, 102, 0.15);
}
.metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #A0AEC0;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* Sidebar styling overrides */
[data-testid="stSidebar"] {
    background-color: #0F121A !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04);
}

/* Custom styled action buttons */
.stButton>button {
    background: linear-gradient(135deg, #00FF66 0%, #00B344 100%) !important;
    color: #000000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(0, 255, 102, 0.12) !important;
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0, 255, 102, 0.25) !important;
}

/* Stop button styling */
.stop-btn>div>button {
    background: linear-gradient(135deg, #FF3B30 0%, #C7000B 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(255, 59, 48, 0.2) !important;
}
.stop-btn>div>button:hover {
    box-shadow: 0 6px 20px rgba(255, 59, 48, 0.35) !important;
}

/* Main premium banner */
.app-banner {
    background: linear-gradient(135deg, #161A24 0%, #0E1117 100%);
    padding: 24px;
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-left: 6px solid #00FF66;
    margin-bottom: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
}
.app-banner h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
}
.app-banner p {
    margin: 6px 0 0 0;
    color: #A0AEC0;
    font-size: 0.95rem;
}

/* Footer elements */
.app-footer {
    text-align: center;
    padding: 24px 10px;
    color: #4A5568;
    font-size: 0.8rem;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
    margin-top: 40px;
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

# Import local modules cleanly
try:
    from vehicle_detector import VehicleDetector
    from utils import draw_bbox, draw_fps, draw_dashboard, check_intersection, CLASS_COLORS
except ImportError as e:
    st.error(f"❌ Failed to import local modules: {e}. Ensure vehicle_detector.py and utils.py are in the same folder.")
    st.stop()

# Cache the YOLO Model loader to load ONLY ONCE and prevent OOM memory leaks
@st.cache_resource
def get_cached_detector(model_path):
    st.sidebar.info(f"⏳ Loading and caching YOLO model `{model_path}`...")
    print(f"[DEBUG] Loading YOLO model from path: {model_path}", file=sys.stderr)
    return VehicleDetector(model_path=model_path)

def find_models():
    """
    Locates YOLOv8 model files dynamically inside project directory.
    """
    local_pt = glob.glob("*.pt")
    parent_pt = glob.glob("../*.pt") + glob.glob("../../*.pt")
    all_pt = list(set(local_pt + [os.path.basename(p) for p in parent_pt]))
    
    standards = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    for s in standards:
        if s not in all_pt:
            all_pt.append(s)
    all_pt.sort()
    return all_pt

# ----------------- SESSION STATE STABILITY MANAGER -----------------
# Explicitly initialize stable session variables to manage reruns and stop commands
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "process_complete" not in st.session_state:
    st.session_state.process_complete = False
if "report_data" not in st.session_state:
    st.session_state.report_data = []
if "final_counts" not in st.session_state:
    st.session_state.final_counts = {}
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "temp_raw_video_path" not in st.session_state:
    st.session_state.temp_raw_video_path = None

def cleanup_temp_files():
    """
    Safely clean up temporary files to avoid disk overflow.
    """
    if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
        try:
            os.remove(st.session_state.processed_video_path)
            print(f"[DEBUG] Deleted old processed video: {st.session_state.processed_video_path}", file=sys.stderr)
        except:
            pass
    if st.session_state.temp_raw_video_path and os.path.exists(st.session_state.temp_raw_video_path):
        try:
            os.remove(st.session_state.temp_raw_video_path)
            print(f"[DEBUG] Deleted old raw video: {st.session_state.temp_raw_video_path}", file=sys.stderr)
        except:
            pass
    st.session_state.processed_video_path = None
    st.session_state.temp_raw_video_path = None

def toggle_processing():
    """
    Thread-safe state toggler callback for Start/Stop buttons.
    """
    if not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.stop_requested = False
        st.session_state.process_complete = False
        st.session_state.report_data = []
        st.session_state.final_counts = {}
    else:
        st.session_state.is_processing = False
        st.session_state.stop_requested = True

# Main Layout Banner
st.markdown("""
<div class="app-banner">
    <h1>AI Traffic Monitor & Vehicle Counting Suite</h1>
    <p>A production-ready YOLOv8 computer vision dashboard for traffic analytics, flow optimization, and detailed audit reports.</p>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS (WIDGET DISABLING MECHANISM) -----------------
# All control sliders are dynamically disabled when processing is active
# This locks the state and prevents users from interrupting the video processing stream.
disabled_during_run = st.session_state.is_processing

st.sidebar.markdown("### ⚙️ Detection Settings")

available_models = find_models()
selected_model = st.sidebar.selectbox(
    "Choose YOLOv8 Model",
    options=available_models,
    index=available_models.index("yolov8m.pt") if "yolov8m.pt" in available_models else 0,
    disabled=disabled_during_run,
    help="If selected model is not local, YOLO will automatically download it."
)

st.sidebar.markdown("### 🎚️ Tracker Thresholds")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.00, 0.15, 0.05, disabled=disabled_during_run)
iou_threshold = st.sidebar.slider("IOU Tracker Threshold", 0.10, 0.95, 0.50, 0.05, disabled=disabled_during_run)

st.sidebar.markdown("### 📏 Crossing Gates Y-Coords (640x360 scale)")
entry_line_y = st.sidebar.slider("Entry Gate (Yellow)", 10, 350, 150, 5, disabled=disabled_during_run)
exit_line_y = st.sidebar.slider("Exit Gate (Magenta)", 10, 350, 250, 5, disabled=disabled_during_run)

st.sidebar.markdown("### ⚡ Inference Optimization")
frame_skipping = st.sidebar.slider(
    "Frame Skipping Factor",
    min_value=1,
    max_value=5,
    value=3,
    disabled=disabled_during_run,
    help="1 = process all frames. 2 = process alternate frames, etc. Drastically speeds up Streamlit Cloud."
)

imgsz_selection = st.sidebar.selectbox(
    "YOLO Inference Size (imgsz)",
    options=[160, 224, 320, 416, 512, 640],
    index=2, # default to 320
    disabled=disabled_during_run,
    help="Smaller size drops computations by 75%+, drastically boosting speed on CPU environments."
)

# Start/Stop Button with Callback
st.sidebar.markdown("---")
if not st.session_state.is_processing:
    # Enabled only if video file is loaded
    uploaded_file_present = st.session_state.temp_raw_video_path is not None
    st.sidebar.button("🚀 Start Detection", on_click=toggle_processing, disabled=not uploaded_file_present)
else:
    # Custom colored stop button class injected from CSS
    st.sidebar.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    st.sidebar.button("🛑 Stop Detection", on_click=toggle_processing)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ----------------- TOP SECTION: VIDEO UPLOADER & INFO CARD -----------------
st.markdown("### 📂 Video Upload Panel")
uploaded_file = st.file_uploader("Upload a traffic video file to begin (MP4, AVI, MOV, MKV)", type=["mp4", "avi", "mov", "mkv"])

# Upload Handling with clean cache manager
if uploaded_file is not None:
    # Check if this is a brand new file
    temp_dir = tempfile.gettempdir()
    # Unique tag to match filename but stay safe
    safe_filename = "".join([c for c in uploaded_file.name if c.isalnum() or c in ['.', '_', '-']])
    proposed_path = os.path.join(temp_dir, f"raw_{safe_filename}")
    
    if st.session_state.temp_raw_video_path != proposed_path:
        cleanup_temp_files() # Flush old temp structures
        with open(proposed_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.temp_raw_video_path = proposed_path
        st.session_state.process_complete = False
        st.session_state.report_data = []
        st.session_state.final_counts = {}
        print(f"[DEBUG] Stored uploaded file at: {proposed_path}", file=sys.stderr)

# Draw Video Info Card if file is successfully stored
if st.session_state.temp_raw_video_path and os.path.exists(st.session_state.temp_raw_video_path):
    cap = cv2.VideoCapture(st.session_state.temp_raw_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    if fps_video <= 0:
        fps_video = 30.0
    duration_sec = total_frames / fps_video if total_frames > 0 else 0.0
    
    # Show Video Info Grid
    inf1, inf2, inf3, inf4, inf5 = st.columns(5)
    inf1.metric("File Name", uploaded_file.name if uploaded_file else "Cached Feed")
    inf2.metric("Resolution", f"{width}x{height}")
    inf3.metric("Video FPS", f"{fps_video:.1f}")
    inf4.metric("Frame Count", f"{total_frames}")
    inf5.metric("Est. Duration", f"{duration_sec:.1f}s")
else:
    st.info("💡 Awaiting video upload. Upload a file above, set your boundaries, and click 'Start Detection'.")

# ----------------- MIDDLE SECTION: CALIBRATION OR BACKGROUND LOADING -----------------
st.markdown("---")

if not st.session_state.is_processing and not st.session_state.process_complete:
    if st.session_state.temp_raw_video_path and os.path.exists(st.session_state.temp_raw_video_path):
        st.markdown("### 📏 Step 2: Calibration & Preview Zone")
        cap = cv2.VideoCapture(st.session_state.temp_raw_video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            frame_resized = cv2.resize(first_frame, (640, 360))
            # Overlay entry/exit lines on static preview
            cv2.line(frame_resized, (0, entry_line_y), (640, entry_line_y), (0, 255, 255), 2) # Yellow
            cv2.line(frame_resized, (0, exit_line_y), (640, exit_line_y), (255, 0, 255), 2) # Magenta
            cv2.putText(frame_resized, f"ENTRY ZONE (Y={entry_line_y})", (20, entry_line_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame_resized, f"EXIT ZONE (Y={exit_line_y})", (20, exit_line_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            
            col_pre_l, col_pre_r = st.columns([7, 3])
            with col_pre_l:
                st.image(frame_resized, channels="BGR", use_container_width=True)
                st.caption("📏 **Calibration Preview**: Yellow Line = Entry Gate, Magenta Line = Exit Gate. Adjust sliders in sidebar.")
            with col_pre_r:
                st.markdown("""
                <div style="background: rgba(30, 36, 50, 0.3); padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <h4 style="margin-top:0; color: #00FF66;">💡 Calibration Guidelines</h4>
                    <p style="font-size: 0.9rem; color: #A0AEC0;">
                        Use the <b>Crossing Gates Y-Coords</b> sliders in the sidebar to adjust boundaries:
                    </p>
                    <ul style="font-size: 0.85rem; color: #A0AEC0; padding-left: 20px;">
                        <li><b>Yellow Line</b>: Entry boundary where vehicle tracking registers.</li>
                        <li><b>Magenta Line</b>: Exit boundary. Once a vehicle crosses this boundary, transit duration is logged and counts increment.</li>
                        <li>Set <b>Frame Skipping</b> and <b>YOLO Inference Size</b> to optimize performance.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Could not read frame from video. File might be corrupted.")
    else:
        st.info("💡 Awaiting video upload. Upload a file above, set your boundaries, and click 'Start Detection'.")

elif st.session_state.is_processing:
    st.markdown("### ⚙️ Video Processing in Progress...")
    
    # Elegant, minimal background progress panel
    progress_container = st.container()
    with progress_container:
        st.spinner("YOLOv8 offline tracker is running at high speed...")
        
        col_prog_l, col_prog_r = st.columns([8, 2])
        with col_prog_l:
            progress_bar = st.progress(0.0)
            timer_text = st.empty()
        with col_prog_r:
            status_text = st.empty()

# ----------------- MAIN VIDEO PROCESSING LOOP (OFFLINE) -----------------
if st.session_state.is_processing and st.session_state.temp_raw_video_path:
    # 1. Cache load detector safely
    try:
        detector = get_cached_detector(selected_model)
    except Exception as e:
        st.error(f"❌ Failed to load YOLO model: {e}")
        st.session_state.is_processing = False
        st.stop()
        
    # 2. Establish unique output paths to prevent lock collisions
    unique_id = uuid.uuid4().hex[:8]
    temp_out_path = os.path.join(tempfile.gettempdir(), f"processed_{unique_id}.mp4")
    
    cap = cv2.VideoCapture(st.session_state.temp_raw_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0
        
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Physically resize target video frames to 640x360 for high efficiency
    out = cv2.VideoWriter(temp_out_path, fourcc, fps_video / frame_skipping, (640, 360))
    
    # 3. Running loop states
    track_history = {}
    track_last_seen = {} # track_id -> frame_idx to purge old inactive vehicles
    active_vehicles = {} # track_id -> entry_time
    counted_ids = set()
    local_logs = []
    
    local_counts = {
        "Car": 0,
        "Bike": 0,
        "Bus": 0,
        "Truck": 0,
        "Auto": 0,
        "Others": 0
    }
    
    # Line endpoints scaled to 640x360
    line1_pt1, line1_pt2 = (0, entry_line_y), (640, entry_line_y)
    line2_pt1, line2_pt2 = (0, exit_line_y), (640, exit_line_y)
    
    frame_idx = 0
    start_time = time.time()
    prev_time = start_time
    
    print(f"[DEBUG] Started offline detection. Resizing to 640x360, imgsz={imgsz_selection}. Output: {temp_out_path}", file=sys.stderr)
    
    try:
        while cap.isOpened():
            # Check if user requested stop via sidebar callback
            if st.session_state.stop_requested:
                print("[DEBUG] Detection loop interrupted by user request.", file=sys.stderr)
                break
                
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Frame skipping logic
            if frame_idx % frame_skipping != 0:
                continue
                
            # Physically resize frame down to 640x360 to optimize processing overhead
            frame_resized = cv2.resize(frame, (640, 360))
            
            # Run YOLO track with user thresholds and optimized imgsz
            detections = detector.detect_and_track(frame_resized, conf=conf_threshold, iou=iou_threshold, imgsz=imgsz_selection)
            
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
                track_last_seen[track_id] = frame_idx
                
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)
                    
                if len(track_history[track_id]) >= 2:
                    prev_pt = track_history[track_id][-2]
                    curr_pt = track_history[track_id][-1]
                    
                    # Crossing Line 1 (Entry Gate)
                    if track_id not in active_vehicles and track_id not in counted_ids:
                        if check_intersection(prev_pt, curr_pt, line1_pt1, line1_pt2):
                            active_vehicles[track_id] = current_time_str
                            color_line1 = (0, 255, 0)
                            
                    # Crossing Line 2 (Exit Gate)
                    elif track_id in active_vehicles:
                        if check_intersection(prev_pt, curr_pt, line2_pt1, line2_pt2):
                            entry_time = active_vehicles.pop(track_id)
                            counted_ids.add(track_id)
                            
                            # Immediately remove tracking history and trails upon crossing exit line
                            track_history.pop(track_id, None)
                            track_last_seen.pop(track_id, None)
                            
                            # Increment category count
                            if label in local_counts:
                                local_counts[label] += 1
                            else:
                                local_counts["Others"] += 1
                                
                            # Calculate crossing duration
                            try:
                                t1 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S.%f")
                                t2 = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f")
                                duration = (t2 - t1).total_seconds()
                            except:
                                duration = 0.0
                                
                            local_logs.append({
                                "Vehicle ID": track_id,
                                "Class": label,
                                "Entry Time": entry_time.split()[1],
                                "Exit Time": current_time_str.split()[1],
                                "Duration (s)": round(duration, 2)
                            })
                            color_line2 = (0, 255, 0)
                            
            # Purge inactive vehicle IDs from memory to prevent memory buildup
            # If not detected for 30 consecutive frames, safely remove
            inactive_threshold = 30
            for tid in list(track_history.keys()):
                if frame_idx - track_last_seen.get(tid, 0) > inactive_threshold:
                    track_history.pop(tid, None)
                    active_vehicles.pop(tid, None)
                    track_last_seen.pop(tid, None)
                            
            # Boundaries overlays & visual trail
            for track_id, pts in track_history.items():
                # Only draw trails for vehicles that haven't crossed the exit line yet
                if track_id not in counted_ids:
                    # Draw premium blue tracking trail (BGR format)
                    for i in range(1, len(pts)):
                        cv2.line(frame_resized, pts[i-1], pts[i], (255, 0, 0), 2)
                        
            # Dynamic visual rendering overlays for current detections
            for det in detections:
                box = det["box"]
                label = det["label"]
                conf = det["conf"]
                track_id = det["track_id"]
                # Only render bounding box for vehicles that haven't crossed the exit line yet
                if track_id not in counted_ids:
                    bbox_color = (0, 255, 0) if track_id in active_vehicles else None
                    draw_bbox(frame_resized, box, label, conf, track_id=track_id, color=bbox_color)
                        
            # Boundaries overlays
            cv2.line(frame_resized, line1_pt1, line1_pt2, color_line1, 2)
            cv2.line(frame_resized, line2_pt1, line2_pt2, color_line2, 2)
            cv2.putText(frame_resized, "ENTRY GATE", (10, entry_line_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_line1, 1)
            cv2.putText(frame_resized, "EXIT GATE", (10, exit_line_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_line2, 1)
            
            # FPS Calculation (average/current frame rate on visual output)
            curr_time_fps = time.time()
            fps = 1.0 / (curr_time_fps - prev_time) if (curr_time_fps - prev_time) > 0 else 0.0
            prev_time = curr_time_fps
            draw_fps(frame_resized, fps)
            
            # Draw beautiful telemetry dashboard directly on the frame!
            draw_dashboard(frame_resized, local_counts, len(counted_ids), len(active_vehicles))
            
            # Write annotated frame to video file
            out.write(frame_resized)
            
            # Sparse updates to Streamlit socket interface (drastically boosts processing speed)
            if frame_idx % 15 == 0 or frame_idx == total_frames:
                pct = min(frame_idx / total_frames, 1.0)
                progress_bar.progress(pct)
                elapsed = time.time() - start_time
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0.0
                status_text.markdown(f"**Progress**: {int(pct*100)}%")
                timer_text.markdown(f"⏱️ **Elapsed**: {elapsed:.1f}s | ⚡ **Average Speed**: {avg_fps:.1f} FPS | **Frame**: {frame_idx}/{total_frames}")
                
    except Exception as run_err:
        print(f"[ERROR] Exception during tracking execution: {run_err}", file=sys.stderr)
        st.error(f"⚠️ Silent crash prevented! Error: {run_err}")
        
    finally:
        # Guarantee resources are fully closed and released!
        cap.release()
        out.release()
        
        # Save variables to stable session states
        st.session_state.is_processing = False
        st.session_state.process_complete = True
        st.session_state.report_data = local_logs
        st.session_state.final_counts = local_counts
        st.session_state.processed_video_path = temp_out_path
        
        print(f"[DEBUG] Finished loop. Releasing resources. Process complete.", file=sys.stderr)
        st.rerun()

# ----------------- BOTTOM SECTION: TABS FOR RESULTS & DOWNLOADS -----------------
# This bottom section stays hidden until the first video is successfully fully processed
if st.session_state.process_complete and not st.session_state.is_processing and len(st.session_state.final_counts) > 0:
    st.markdown("---")
    st.markdown("### 🏆 Step 3: Traffic Analysis & Deep Analytics Suite")
    
    logs = st.session_state.report_data
    counts = st.session_state.final_counts
    total_audited = sum(counts.values())
    
    # 7 premium glassmorphic statistics cards
    c_tot, c_car, c_bike, c_bus, c_truck, c_auto, c_oth = st.columns(7)
    c_tot.markdown(f'<div class="metric-card"><div class="metric-value">{total_audited}</div><div class="metric-label">TOTAL COUNT</div></div>', unsafe_allow_html=True)
    c_car.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Car", 0)}</div><div class="metric-label">🚗 Cars</div></div>', unsafe_allow_html=True)
    c_bike.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Bike", 0)}</div><div class="metric-label">🏍️ Bikes</div></div>', unsafe_allow_html=True)
    c_bus.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Bus", 0)}</div><div class="metric-label">🚌 Buses</div></div>', unsafe_allow_html=True)
    c_truck.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Truck", 0)}</div><div class="metric-label">🚛 Trucks</div></div>', unsafe_allow_html=True)
    c_auto.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Auto", 0)}</div><div class="metric-label">🛺 Autos</div></div>', unsafe_allow_html=True)
    c_oth.markdown(f'<div class="metric-card"><div class="metric-value">{counts.get("Others", 0)}</div><div class="metric-label">📦 Others</div></div>', unsafe_allow_html=True)
    
    col_res_l, col_res_r = st.columns([6, 4])
    with col_res_l:
        st.markdown("#### 🎬 Play Processed Output Video")
        if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
            st.video(st.session_state.processed_video_path)
            st.caption("ℹ️ Download options are available in the File Export Center tab below.")
        else:
            st.error("Processed video file was not found.")
            
    with col_res_r:
        st.markdown("#### 📊 Interactive Flow Distribution")
        if total_audited > 0:
            # Plotly Pie Chart
            pie_df = pd.DataFrame({"Category": list(counts.keys()), "Audit Count": list(counts.values())})
            pie_df = pie_df[pie_df["Audit Count"] > 0]
            fig_pie = px.pie(
                pie_df, 
                values="Audit Count", 
                names="Category", 
                title="Traffic Share Distribution",
                color="Category",
                color_discrete_map={
                    "Car": "#2596be", "Bike": "#ffa500", "Bus": "#00ff00",
                    "Truck": "#ff00ff", "Auto": "#f7d038", "Others": "#808080"
                },
                hole=0.4
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#ffffff')),
                title_font=dict(color='#ffffff'),
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Awaiting tracking timeline data to compile distribution.")
            
    # Tabs below the main player layout
    st.markdown("### 📥 Reports & Data Exports")
    tab_logs, tab_charts, tab_csv, tab_downloads = st.tabs([
        "📜 Crossing Audit Logs", 
        "📈 Flow Density Timeline", 
        "📋 CSV Spreadsheet Preview", 
        "📥 File Export Center"
    ])
    
    # TAB 1: Crossing Logs
    with tab_logs:
        st.markdown("#### Live Transit Crossing Timeline")
        if len(logs) > 0:
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ No vehicles crossed both lines during this video stream.")
            
    # TAB 2: Flow Density Timeline
    with tab_charts:
        st.markdown("#### Flow Density Timeline")
        if len(logs) > 0:
            flow_df = pd.DataFrame(logs)
            flow_df["Secs"] = flow_df["Exit Time"].apply(lambda x: sum(int(t) * 60**i for i, t in enumerate(reversed(x.split(':')))))
            flow_df["Interval"] = (flow_df["Secs"] // 5) * 5
            grouped_flow = flow_df.groupby("Interval").size().reset_index(name="Count")
            
            fig_line = px.line(
                grouped_flow, 
                x="Interval", 
                y="Count", 
                markers=True,
                title="Flow Density Rate (Vehicles per 5s)",
                labels={"Interval": "Transit Timeline (seconds)", "Count": "Vehicles crossing"}
            )
            fig_line.update_traces(line_color="#00FF66", marker=dict(size=6, color="#ffffff"))
            fig_line.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#888888')),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#888888')),
                title_font=dict(color='#ffffff'),
                margin=dict(t=30, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.caption("Not enough dataset points to chart timelines.")

    # TAB 3: CSV Preview
    with tab_csv:
        st.markdown("#### Dynamic CSV Report Audit Sheet")
        # In-memory CSV buffer compiler
        csv_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='w')
        writer = csv.writer(csv_buffer)
        writer.writerow(["Vehicle ID", "Class", "Entry Time", "Exit Time", "Transit Duration (sec)"])
        for item in logs:
            writer.writerow([item["Vehicle ID"], item["Class"], item["Entry Time"], item["Exit Time"], item["Duration (s)"]])
            
        writer.writerow([])
        writer.writerow(["--- SUMMARY AUDIT STATS ---"])
        for k, v in counts.items():
            writer.writerow([k, f"Total counted: {v}"])
        writer.writerow(["TOTAL AUDITED", total_audited])
        csv_buffer.close()
        
        # Read back CSV
        with open(csv_buffer.name, 'r') as f:
            csv_preview_str = f.read()
            
        st.text_area("Live CSV Preview", value=csv_preview_str, height=200, disabled=True)
        
    # TAB 4: Exports Download Center
    with tab_downloads:
        st.markdown("#### Export Audit Records")
        
        down1, down2 = st.columns(2)
        
        with down1:
            st.download_button(
                label="📥 Download Detailed CSV Audit Report",
                data=csv_preview_str,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            st.caption("Exports professional spreadsheet format for data integration.")
            
        with down2:
            if st.session_state.processed_video_path and os.path.exists(st.session_state.processed_video_path):
                with open(st.session_state.processed_video_path, 'rb') as vf:
                    video_bytes = vf.read()
                    
                st.download_button(
                    label="📥 Download Processed Annotated Video",
                    data=video_bytes,
                    file_name="traffic_detection_output.mp4",
                    mime="video/mp4"
                )
                st.caption("Downloads completed MP4 video containing drawing boxes and crossing visual markers.")
            else:
                st.warning("Video file not found or failed to compile correctly.")

# Footer info panel
st.markdown("""
<div class="app-footer">
    AI Traffic Monitor &copy; 2026. Powered by <a href="https://ultralytics.com/" target="_blank">Ultralytics YOLOv8</a> & <a href="https://streamlit.io/" target="_blank">Streamlit</a>.<br>
    Ready for deployment to <b>GitHub</b> and <b>Streamlit Community Cloud</b>.
</div>
""", unsafe_allow_html=True)
