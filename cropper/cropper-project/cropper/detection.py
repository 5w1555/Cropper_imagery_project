import numpy as np
import torch

def simple_nms(boxes, scores, threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i.item())

        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])

        w = torch.maximum(torch.tensor(0), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0), yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        indices = torch.nonzero(iou <= threshold).squeeze()
        order = order[indices + 1]

    return torch.tensor(keep, dtype=torch.int64)

def get_face_and_landmarks(image):
    # Placeholder function for face detection and landmark extraction
    # This should be replaced with actual face detection logic
    return [], {}

def is_frontal_face(landmarks):
    # Simple heuristic to determine if the face is frontal based on landmarks
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    
    if left_eye and right_eye:
        return abs(left_eye[0] - right_eye[0]) < 10  # Example threshold
    return False