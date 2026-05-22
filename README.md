# AI Traffic Monitor & Vehicle Counting Suite 🚗📊

A professional, production-grade computer vision dashboard designed to monitor traffic flows, detect and track vehicles, and generate detailed CSV audit reports using **YOLOv8** and **Streamlit**.

This suite offers two operating modes:
1. **Interactive Streamlit Web Dashboard** (`app.py`): A modern, minimalist glassmorphic dark-mode web application featuring real-time detection streams, interactive sliders for calibrating dual crossing lines, dynamic Plotly visual analytics, and CSV report exports.
2. **Local OpenCV Pipeline** (`main.py`): An efficient command-line utility with mouse-drawn trigger zones.

---

## 🌟 Key Features

- **Object Tracking**: Built-in tracking using BOT-SORT trackers to track multiple vehicle paths continuously.
- **Dynamic Line Calibration**: Live sliders in the Streamlit sidebar allow real-time repositioning of entry/exit boundary lines over a static video preview.
- **Vehicle Classification**: Automatically detects and separates vehicles into:
  - **Car**
  - **Bike** (Motorcycle, Bicycle, Scooter, Person)
  - **Bus**
  - **Truck** (Truck/SCV/LCV)
  - **Auto** (Auto-rickshaws, Tuk-tuks)
  - **Others** (custom detections)
- **High-Performance CPU Optimizations**: Frame-skipping parameter controls (1x to 5x alternate processing) to maintain fluid tracking even on CPU-constrained environments like Streamlit Community Cloud.
- **Interactive Visual Dashboard**:
  - Live metric cards (FPS, Active tracker count, Total counted, Elapsed run-time).
  - Real-time Plotly charts displaying vehicle class ratios.
  - Interactive table of recent counts.
- **Complete Analytics Suite**:
  - Pie charts of overall traffic volume distribution.
  - Temporal traffic density flow lines (Traffic Volume per 5-second intervals).
  - Instant CSV audit download matching corporate standards.
  - Streamlit video downloader for processed and annotated footage.

---

## 📁 Repository Structure

```text
├── .streamlit/
│   └── config.toml          # Premium custom dark-theme rules
├── .gitignore               # Ensures lightweight repository (excl. large videos/pt models)
├── README.md                # Detailed user manual (this file)
├── requirements.txt         # Package dependencies (optimized for headless servers)
├── app.py                   # Main Streamlit Web Application Dashboard [NEW]
├── main.py                  # CLI implementation with OpenCV mouse clicks
├── vehicle_detector.py      # Core wrapper class for the YOLOv8 tracking library
├── utils.py                 # Core graphic functions (bounding box drawing, overlay panel)
├── classes.txt              # Standard target YOLO COCO class lists
└── config.yaml              # Custom model definitions and YAML dataset bounds
```

---

## 💻 Local Installation & Setup

Ensure you have **Python 3.8+** installed on your system.

### 1. Clone or Locate Project Folder
Open your terminal/command prompt and navigate to the project directory:
```bash
cd S:\DATS\Trail\vehicle_detection_project
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate an isolated environment to prevent library version conflicts:
```bash
# Create Virtual Environment
python -m venv venv

# Activate on Windows (PowerShell/CMD)
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Package Dependencies
Install the required packages. Note that `requirements.txt` relies on `opencv-python-headless` for server compatibility. For local GUI windows (needed by `main.py` but not `app.py`), you can use either headless or standard OpenCV.
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
This will automatically open the app in your default browser at `http://localhost:8501`.

---

## 🌐 Deploying to Streamlit Community Cloud

Streamlit Community Cloud is a free, fast platform to host your interactive dashboards directly from GitHub.

### Step 1: Push Project to GitHub
Follow the instructions in the next section to push your files to your GitHub repository `https://github.com/rebbasivesh/vehicalcounttest.git`.

### Step 2: Set Up Hosting
1. Visit [Streamlit Share](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Fill out the application form:
   - **Repository**: `rebbasivesh/vehicalcounttest`
   - **Branch**: `main` (or your active branch)
   - **Main file path**: `app.py`
4. Click **Deploy!** 🚀

Streamlit Cloud will automatically build your python environment using the `requirements.txt` file and spin up your application.

---

## 🐙 GitHub Upload Instructions

Use the following commands to upload the files under your project folder to your remote repository `https://github.com/rebbasivesh/vehicalcounttest.git`.

> [!WARNING]
> Ensure you run this inside the `vehicle_detection_project` directory so that large raw video recordings (`.mp4`, `.zip`) and heavy YOLO model checkpoints (`.pt`) are automatically filtered out by our configured `.gitignore`.

```bash
# 1. Initialize empty Git repository
git init

# 2. Add remote origin repository link
git remote add origin https://github.com/rebbasivesh/vehicalcounttest.git

# 3. Rename default branch to main
git branch -M main

# 4. Stage all project files (except ignored ones)
git add .

# 5. Commit files to history
git commit -m "feat: implement premium Streamlit dashboard with interactive calibration, dynamic analytics, and headless cloud support"

# 6. Push code to GitHub
git push -u origin main
```

---

## 🛠️ Calibration & Parameters Quick-Guide

1. **Model Selection**: Standard pre-trained models will download automatically from Ultralytics servers on the first run.
2. **Sliders**:
   - Move **Entry Line** (Yellow) and **Exit Line** (Magenta) using the sidebar sliders. Align them such that incoming traffic crosses the Yellow line first (enters zone), and then the Magenta line (is counted/classified).
   - Use **Confidence Threshold** to filter out low-confidence background noise (default `0.15` works best for average angles).
   - Increase **Frame Skipping** (e.g. to `2` or `3`) if you are deploying to Streamlit Cloud or have standard CPU setups; this maintains vehicle tracking while bypassing alternate frames, accelerating processing up to 300%.
