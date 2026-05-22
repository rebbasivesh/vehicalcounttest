import cv2
import time
import os
import csv
from datetime import datetime
from vehicle_detector import VehicleDetector
from utils import draw_bbox, draw_fps, draw_dashboard, check_intersection

# Global variables for trigger line drawing
trigger_points = []
drawing = False

def draw_line(event, x, y, flags, param):
    """
    Mouse callback function to draw the trigger lines.
    Requires exactly 4 points (2 lines).
    """
    global trigger_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(trigger_points) >= 4:
            trigger_points = [] # Reset if already drawn
        trigger_points.append((x, y))

def main():
    global trigger_points
    
    # Configuration
    video_path = r"S:\DATS\Trail\2e115b76df11491f0a5898d3cefab132.mp4"
    
    user_input = input(f"Enter video path (or press Enter to use default '{video_path}'): ")
    if user_input.strip() != "":
        video_path = user_input.strip()

    if not os.path.exists(video_path):
        print(f"\n[!] Error: Could not find video file at '{video_path}'")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n[!] Error: Failed to open video file '{video_path}'")
        return

    # Read first frame to let user draw the trigger lines
    ret, first_frame = cap.read()
    if not ret:
        print("\n[!] Error: Cannot read the video.")
        return
        
    first_frame = cv2.resize(first_frame, (1020, 600))
    
    cv2.namedWindow("Draw Trigger Lines")
    cv2.setMouseCallback("Draw Trigger Lines", draw_line)
    
    print("\n--- DRAW DUAL TRIGGER LINES ---")
    print("1. Click 2 points for Line 1 (Entry / Detection).")
    print("2. Click 2 points for Line 2 (Exit / Classification).")
    print("3. Press 'Enter' when you are happy with both lines.")
    print("4. Press 'r' to reset.")
    
    while True:
        temp_frame = first_frame.copy()
        
        # Instructions
        cv2.putText(temp_frame, "Draw Line 1 (Entry) then Line 2 (Exit). Click 4 points total.", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(temp_frame, "Press 'Enter' to start, 'r' to reset.", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw points and lines
        for i, pt in enumerate(trigger_points):
            color = (0, 255, 255) if i < 2 else (255, 0, 255) # Yellow for Line 1, Magenta for Line 2
            cv2.circle(temp_frame, pt, 5, color, -1)
            
        if len(trigger_points) >= 2:
            cv2.line(temp_frame, trigger_points[0], trigger_points[1], (0, 255, 255), 2)
            cv2.putText(temp_frame, "Line 1 (Entry)", (trigger_points[0][0], trigger_points[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        if len(trigger_points) == 4:
            cv2.line(temp_frame, trigger_points[2], trigger_points[3], (255, 0, 255), 2)
            cv2.putText(temp_frame, "Line 2 (Exit)", (trigger_points[2][0], trigger_points[2][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        cv2.imshow("Draw Trigger Lines", temp_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13: # Enter key
            if len(trigger_points) == 4:
                break
            else:
                print("Please draw exactly 4 points (2 lines) first!")
        elif key == ord('r'):
            trigger_points = []
            
    cv2.destroyWindow("Draw Trigger Lines")

    line1_pt1, line1_pt2 = trigger_points[0], trigger_points[1]
    line2_pt1, line2_pt2 = trigger_points[2], trigger_points[3]

    # Setup Video Output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = 'output_processed.mp4'
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (1020, 600))

    print("\nLoading YOLOv8 model...")
    detector = VehicleDetector(model_path="yolov8m.pt")
    
    track_history = {}
    active_vehicles = {} # track_id -> entry_timestamp (str)
    counted_ids = set()
    vehicle_logs = [] # List of dicts for CSV: {ID, Class, Entry Time, Exit Time, Duration}
    
    counts = {
        "2 Wheeler": 0,
        "Car": 0,
        "Bus": 0,
        "Truck/SCV/LCV": 0
    }
    
    prev_time = time.time()
    print("\nProcessing... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video reached.")
            break
            
        frame = cv2.resize(frame, (1020, 600))
        
        detections = detector.detect_and_track(frame)
        
        # Default line colors
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
                
                # Check intersection with Line 1 (Entry)
                if track_id not in active_vehicles and track_id not in counted_ids:
                    if check_intersection(prev_pt, curr_pt, line1_pt1, line1_pt2):
                        active_vehicles[track_id] = current_time_str
                        color_line1 = (0, 255, 0) # Flash Green
                
                # Check intersection with Line 2 (Exit / Classification)
                elif track_id in active_vehicles:
                    if check_intersection(prev_pt, curr_pt, line2_pt1, line2_pt2):
                        entry_time = active_vehicles.pop(track_id)
                        counted_ids.add(track_id)
                        
                        if label in counts:
                            counts[label] += 1
                            
                        # Log to CSV data
                        try:
                            t1 = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S.%f")
                            t2 = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M:%S.%f")
                            duration = (t2 - t1).total_seconds()
                        except:
                            duration = 0.0
                            
                        vehicle_logs.append({
                            "Vehicle ID": track_id,
                            "Class": label,
                            "Entry Time (Line 1)": entry_time,
                            "Exit Time (Line 2)": current_time_str,
                            "Duration (sec)": round(duration, 2)
                        })
                        
                        color_line2 = (0, 255, 0) # Flash Green
            
            # Visibility logic: STRICTLY ignore vehicles before Entry line
            if track_id in active_vehicles or track_id in counted_ids:
                # Highlight active vehicles in green in the monitoring zone
                bbox_color = (0, 255, 0) if track_id in active_vehicles else None
                
                draw_bbox(frame, box, label, conf, track_id=track_id, color=bbox_color)
                
                pts = track_history[track_id]
                # To make it even cleaner, we can optionally only draw the trail from the entry point
                # But for now, just drawing the history is fine as it shows the path
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], (255, 255, 0), 2)
                
        # Draw lines
        cv2.line(frame, line1_pt1, line1_pt2, color_line1, 3)
        cv2.line(frame, line2_pt1, line2_pt2, color_line2, 3)
        cv2.putText(frame, "ENTRY", (line1_pt1[0], line1_pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line1, 2)
        cv2.putText(frame, "EXIT", (line2_pt1[0], line2_pt1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line2, 2)
        
        # Calculate FPS
        curr_time_fps = time.time()
        fps = 1 / (curr_time_fps - prev_time) if (curr_time_fps - prev_time) > 0 else 0
        prev_time = curr_time_fps
        
        draw_fps(frame, fps)
        
        total_count = sum(counts.values())
        active_count = len(active_vehicles)
        draw_dashboard(frame, counts, total_count, active_count)
        
        cv2.imshow("Intelligent Traffic Monitor", frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting program by user request.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save CSV Report
    print("\nSaving Detailed CSV Report...")
    with open('report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Vehicle ID", "Class", "Entry Time (Line 1)", "Exit Time (Line 2)", "Duration (sec)"])
        for log in vehicle_logs:
            writer.writerow([log["Vehicle ID"], log["Class"], log["Entry Time (Line 1)"], log["Exit Time (Line 2)"], log["Duration (sec)"]])
            
        writer.writerow([])
        writer.writerow(["--- SUMMARY ---", "", "", "", ""])
        writer.writerow(["Category", "Total Count", "", "", ""])
        for k, v in counts.items():
            writer.writerow([k, v, "", "", ""])
        writer.writerow(["TOTAL", total_count, "", "", ""])
        
    print("Report saved as 'report.csv'")
    print("Processed video saved as 'output_processed.mp4'")
    print("Processing Complete!")

if __name__ == "__main__":
    main()