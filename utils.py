import cv2
import numpy as np

# A gorgeous premium color palette for different vehicle types (BGR format)
CLASS_COLORS = {
    "Car": (255, 255, 0),          # Cyan
    "Bike": (0, 165, 255),         # Orange
    "Bus": (0, 255, 0),            # Green
    "Truck": (255, 0, 255),        # Magenta
    "Auto": (0, 255, 255),         # Yellow
    "Others": (128, 128, 128),     # Gray
    "Unknown": (128, 128, 128)     # Gray
}

def draw_bbox(frame, box, label, confidence, track_id=None, color=None):
    """
    Draws a stylish bounding box with label, confidence score, and tracking ID.
    """
    if color is None:
        color = CLASS_COLORS.get(label, CLASS_COLORS["Unknown"])
        
    x1, y1, x2, y2 = map(int, box)
    
    # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare text
    text = f"ID:{track_id} {label} {confidence:.2f}" if track_id is not None else f"{label} {confidence:.2f}"
    
    # Get text size to draw background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Draw filled rectangle for text background
    cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1)
    
    # Put text over the filled rectangle (black text for contrast)
    cv2.putText(frame, text, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Optional: Draw a center point for tracking visualization
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, color, -1)

def draw_fps(frame, fps):
    """
    Draws a modern FPS counter on the frame.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def draw_dashboard(frame, counts, total_count, active_count):
    """
    Draws a futuristic semi-transparent dashboard on the top-right.
    """
    h, w = frame.shape[:2]
    
    # Dashboard dimensions optimized for 640x360 and higher resolutions
    dash_w = 180
    dash_h = 220
    margin = 15
    x_offset = w - dash_w - margin
    y_offset = margin
    
    # Create an overlay for transparency
    overlay = frame.copy()
    
    # Draw panel background (Dark grey with 75% opacity)
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + dash_w, y_offset + dash_h), (20, 20, 20), -1)
    # Draw panel border
    cv2.rectangle(overlay, (x_offset, y_offset), (x_offset + dash_w, y_offset + dash_h), (102, 255, 0), 2) # Neon Green border (BGR: 0, 255, 102 -> (102, 255, 0))
    
    # Blend overlay with original frame
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Draw Header
    cv2.putText(frame, "TRAFFIC STATS", (x_offset + 25, y_offset + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 255, 0), 2)
    cv2.line(frame, (x_offset + 10, y_offset + 30), (x_offset + dash_w - 10, y_offset + 30), (102, 255, 0), 1)
    
    # Draw Counts
    y_text = y_offset + 50
    line_spacing = 22
    
    # Active Tracking IDs
    cv2.putText(frame, f"Active IDs: {active_count}", (x_offset + 15, y_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    y_text += line_spacing
    
    # Total Count in bold/larger font
    cv2.putText(frame, f"TOTAL: {total_count}", (x_offset + 15, y_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    y_text += line_spacing + 5
    
    # Individual Counts
    for category, count in counts.items():
        color = CLASS_COLORS.get(category, (255, 255, 255))
        cv2.putText(frame, f"{category}: {count}", (x_offset + 15, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        y_text += line_spacing

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def check_intersection(A, B, C, D):
    """
    Return true if line segments AB and CD intersect.
    A, B are previous and current positions of the vehicle.
    C, D are the endpoints of the trigger line.
    """
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
