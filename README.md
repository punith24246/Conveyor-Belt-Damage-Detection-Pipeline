# Conveyor Belt Damage Detection Pipeline

## Overview
This project implements an end-to-end pipeline for detecting conveyor belt damage, specifically:
- **Scratch defects (surface-level)**
- **Edge damage defects (boundary-level)**

The system processes images captured at different times of the day and generates structured outputs for evaluation using **mF1@0.5–0.95**.

---

## Approach

### 1. Belt ROI Detection (YOLOv8)

- Model: YOLOv8n / YOLOv8n-seg  
- Task: Detect conveyor belt region (ROI)  
- Input: Full image  
- Output: Cropped belt region  

ROI extraction is performed using:
- Segmentation masks (preferred)
- Bounding box fallback

---

## Training Details

- Dataset: YOLO-format belt annotations  
- Train images: 287  
- Validation images: 72  
- Image size: 640  
- Batch size: 16  

### Performance

- Precision: ~0.999  
- Recall: 1.0  
- mAP50: ~0.995  
- mAP50-95: ~0.995  

---

## 2. Defect Detection (Classical CV)

Since defect-level annotations were not provided, defect detection is implemented using classical image processing techniques.

### Scratch Detection

- CLAHE (contrast enhancement)
- Gaussian blur
- Top-hat & black-hat transforms
- Thresholding (Otsu)
- Morphological filtering
- Connected components

**Criteria:**
- High aspect ratio (> 4.0)
- Minimum area filtering
- Interior region only

---

### Edge Damage Detection

- Canny edge detection
- Edge band masking (outer 12% region)
- Morphological operations
- Connected components

**Criteria:**
- Near belt boundaries
- Low compactness OR large area

---

## 3. Classification Logic


```bash
if edge_score > 0.005 → edge_damage
elif scratch_score > 0.002 → scratch
else → normal
```



---

## 4. Post-processing

- Confidence-based filtering  
- Noise removal using area thresholds  
- Bounding box refinement  
- ROI-based processing  

---

## Output Format

### Annotated Image
- Saved as: `<image_name>.jpg`
- Contains bounding boxes and predictions

### JSON File
- Saved as: `<image_name>.json`

Example:
```json
{
  "1": {
    "bbox_coordinates": [x_min, y_min, x_max, y_max]
  }
}
```

### Evaluation Metric
- mF1@0.5–0.95
- IoU-based matching
- TP / FP / FN calculation
- Thresholds: 0.50 → 0.95
- Final score = average F1

 # How to Run
Install dependencies
```bash
pip install ultralytics opencv-python-headless numpy
```

# Run pipeline
```bash
python pipeline.py \
  --image_dir /content/drive/MyDrive/belt_dataset/val/images \
  --output_dir /content/output \
  --roi_model /content/drive/MyDrive/belt_runs/belt_roi_seg/weights/best.pt
```

# Arguments
```bash
--image_dir → Input images
--output_dir → Output folder
--roi_model → Trained YOLO model
```

### Output

- Annotated images in output/
- JSON predictions per image
- ROI visualizations in output/visualizations/
- Important Note
- Dataset contains only belt annotations
- No defect-level labels provided
- Hybrid approach (YOLO + classical CV) used
- Evaluation performed using hidden ground truth









