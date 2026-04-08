import os
import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def clahe_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def get_roi_from_model(model, image):
    results = model.predict(image, conf=0.25, verbose=False)
    if len(results) == 0:
        return None, None

    r = results[0]

    if hasattr(r, "masks") and r.masks is not None and len(r.masks.data) > 0:
        mask = r.masks.data[0].cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None, None
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    if r.boxes is not None and len(r.boxes) > 0:
        box = r.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        crop = image[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    return None, None


def detect_scratch_boxes(roi):
    if roi is None or roi.size == 0:
        return []

    gray = clahe_gray(roi)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.addWeighted(tophat, 1.0, blackhat, 1.0, 0)

    _, th = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=1)

    h, w = th.shape
    border_x = int(0.08 * w)
    border_y = int(0.08 * h)
    mask_inside = np.zeros_like(th)
    mask_inside[border_y:h-border_y, border_x:w-border_x] = 255
    th_inside = cv2.bitwise_and(th, mask_inside)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th_inside, 8)

    boxes = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 20:
            continue
        aspect = max(ww, hh) / (min(ww, hh) + 1e-6)
        if area > 25 and aspect > 4.0:
            boxes.append((x, y, x + ww, y + hh, area / (h * w + 1e-6)))

    return boxes


def detect_edge_damage_boxes(roi):
    if roi is None or roi.size == 0:
        return []

    gray = clahe_gray(roi)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    band = int(0.12 * min(h, w))
    edge_band_mask = np.zeros_like(edges)
    edge_band_mask[:band, :] = 255
    edge_band_mask[h-band:, :] = 255
    edge_band_mask[:, :band] = 255
    edge_band_mask[:, w-band:] = 255

    edge_band = cv2.bitwise_and(edges, edge_band_mask)
    edge_band = cv2.dilate(edge_band, np.ones((3, 3), np.uint8), iterations=1)
    edge_band = cv2.morphologyEx(edge_band, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_band, 8)

    boxes = []
    for i in range(1, num_labels):
        x, y, ww, hh, area = stats[i]
        if area < 30:
            continue
        perimeter_like = 2 * (ww + hh)
        compactness = area / (perimeter_like + 1e-6)
        score = area / (h * w + 1e-6)
        if compactness < 3.5 or area > 120:
            boxes.append((x, y, x + ww, y + hh, score))

    return boxes


def nms_boxes(boxes, iou_thresh=0.3):
    if not boxes:
        return []

    boxes_np = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=float)
    scores = np.array([b[4] for b in boxes], dtype=float)

    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def merge_damage_boxes(roi):
    scratch_boxes = detect_scratch_boxes(roi)
    edge_boxes = detect_edge_damage_boxes(roi)

    filtered = []

    for x1, y1, x2, y2, score in scratch_boxes:
        if score > 0.002:
            filtered.append((x1, y1, x2, y2, score))

    for x1, y1, x2, y2, score in edge_boxes:
        if score > 0.005:
            filtered.append((x1, y1, x2, y2, score))

    return nms_boxes(filtered, iou_thresh=0.3)


def save_json(json_path, boxes):
    result = {}
    for idx, (x1, y1, x2, y2, score) in enumerate(boxes, start=1):
        result[str(idx)] = {
            "bbox_coordinates": [int(x1), int(y1), int(x2), int(y2)]
        }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--roi_model", type=str, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    model = YOLO(args.roi_model)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    for fname in sorted(os.listdir(args.image_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in valid_ext:
            continue

        img_path = os.path.join(args.image_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            continue

        roi, belt_bbox = get_roi_from_model(model, image)

        base_name = os.path.splitext(fname)[0]
        out_img_path = os.path.join(args.output_dir, f"{base_name}.jpg")
        out_json_path = os.path.join(args.output_dir, f"{base_name}.json")

        if roi is None or roi.size == 0:
            cv2.imwrite(out_img_path, image)
            save_json(out_json_path, [])
            continue

        roi_damage_boxes = merge_damage_boxes(roi)

        x_off, y_off = belt_bbox[0], belt_bbox[1]
        full_image_boxes = []
        annotated = image.copy()

        for (x1, y1, x2, y2, score) in roi_damage_boxes:
            gx1, gy1 = x1 + x_off, y1 + y_off
            gx2, gy2 = x2 + x_off, y2 + y_off

            full_image_boxes.append((gx1, gy1, gx2, gy2, score))
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)

        cv2.imwrite(out_img_path, annotated)
        save_json(out_json_path, full_image_boxes)

    print(f"Done. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
