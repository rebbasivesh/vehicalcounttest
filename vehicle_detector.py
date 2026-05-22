import sys

# ----------------- TRACKER DEBBUGGING FALLBACK SYSTEM -----------------
# Dynamically register lapx as lap in sys.modules to prevent the classic
# "No module named 'lap'" error in YOLOv8's internal Hungarian matcher.
try:
    import lap
except ImportError:
    try:
        import lapx as lap
        sys.modules['lap'] = lap  # Inject lapx into standard lap namespace
        print("[DEBUG] Successfully injected lapx as lap wrapper.", file=sys.stderr)
    except ImportError:
        print("[WARNING] Neither 'lap' nor 'lapx' is installed. YOLO tracking might fail.", file=sys.stderr)

from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt"):
        """
        Initializes the YOLOv8 model.
        Automatically downloads the model if it's not found locally.
        """
        self.model = YOLO(model_path)
        
        # YOLOv8 COCO classes for vehicles and riders:
        # 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]
        self.class_names = self.model.names
        
    def map_class_to_category(self, class_id):
        """
        Dynamically maps COCO class IDs or custom trained model class names
        to standard vehicle categories: Car, Bike, Bus, Truck, Auto, Others.
        """
        name = self.class_names.get(class_id, "Unknown").lower()
        
        # String pattern matching for maximum compatibility with custom models
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

    def detect_and_track(self, frame, conf=0.15, iou=0.5, imgsz=320):
        """
        Detects and tracks vehicles in the given frame using built-in tracker.
        Accepts dynamic confidence and IOU thresholds from the UI sliders.
        Supports dynamic imgsz parameter for inference optimization.
        Returns a list of parsed detections with tracking IDs.
        """
        # Run tracking using botsort.yaml tracker with dynamic settings
        results = self.model.track(
            frame, 
            classes=self.vehicle_classes, 
            persist=True, 
            verbose=False, 
            tracker="botsort.yaml", 
            conf=conf,
            iou=iou,
            imgsz=imgsz
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf_score = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    mapped_label = self.map_class_to_category(cls_id)
                    
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": cls_id,
                        "label": mapped_label,
                        "conf": conf_score,
                        "track_id": track_id
                    })
                
        return detections
