from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path="yolov8m.pt"): # Upgraded to YOLOv8 Medium for even better accuracy
        """
        Initializes the YOLOv8 model.
        Automatically downloads the model if it's not found locally.
        """
        self.model = YOLO(model_path)
        
        # YOLOv8 COCO classes for vehicles:
        # 0: person (often riders on scooters/bikes are detected as person)
        # 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [0, 1, 2, 3, 5, 7]
        self.class_names = self.model.names
        
    def map_class_to_category(self, class_id):
        """
        Maps standard COCO class IDs to the requested vehicle categories.
        """
        if class_id in [0, 1, 3]:
            return "2 Wheeler" # Person on a bike/scooter is usually detected as person
        elif class_id == 2:
            return "Car"
        elif class_id == 5:
            return "Bus"
        elif class_id == 7:
            return "Truck/SCV/LCV"
        else:
            return "Unknown"

    def detect_and_track(self, frame):
        """
        Detects and tracks vehicles in the given frame using built-in tracker.
        Returns a list of detections with tracking IDs.
        """
        # Run tracking. Lowered confidence threshold (conf=0.15) to capture tricky detections.
        results = self.model.track(frame, classes=self.vehicle_classes, persist=True, verbose=False, tracker="botsort.yaml", conf=0.15)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    mapped_label = self.map_class_to_category(cls_id)
                    
                    detections.append({
                        "box": (x1, y1, x2, y2),
                        "class_id": cls_id,
                        "label": mapped_label,
                        "conf": conf,
                        "track_id": track_id
                    })
                
        return detections
