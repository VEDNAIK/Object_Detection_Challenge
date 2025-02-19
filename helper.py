# Calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union = box1_area + box2_area - intersection
    return intersection / union

# Merge YOLO and OWL-ViT detections based on IoU threshold
def merge_detections(yolo_detections, dino_detections, iou_threshold=0.5):
    merged_detections = []
    for dino_det in dino_detections:
        match_found = False
        for yolo_det in yolo_detections:
            iou = calculate_iou(dino_det["box"], yolo_det["box"])
            if iou > iou_threshold:
                # Take the higher confidence detection
                merged_detections.append(dino_det if dino_det["confidence"] > yolo_det["confidence"] else yolo_det)
                match_found = True
                break
        if not match_found:
            merged_detections.append(dino_det)
    for yolo_det in yolo_detections:
        if all(calculate_iou(yolo_det["box"], d["box"]) <= iou_threshold for d in merged_detections):
            merged_detections.append(yolo_det)
    return merged_detections