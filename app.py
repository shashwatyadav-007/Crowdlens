"""
CrowdLens — Crowd Density Management System
Backend: Flask + OpenCV + YOLOv8 (large) + NumPy

Detection pipeline:
  1. YOLOv8-large tiled 640px inference (30% overlap, conf=0.10)
  2. Haar upper-body cascade recovers occluded heads YOLO missed
  3. IoU-NMS (0.35) merges duplicates from both detectors
  4. Gaussian KDE density map  →  JET colormap heatmap
  5. NumPy spatial grid binning → cell counts

Run: python app.py
"""

import os, time, base64, uuid, traceback, threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("uploads")
OUTPUT_FOLDER = Path("outputs")
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# ─── Model loaders ────────────────────────────────────────────────────────────
_yolo    = None
_cascade = None
_lock    = threading.Lock()

def get_yolo():
    global _yolo
    if _yolo is None:
        with _lock:
            if _yolo is None:
                try:
                    from ultralytics import YOLO
                    for weights in ("yolov8l.pt", "yolov8m.pt", "yolov8s.pt", "yolov8n.pt"):
                        try:
                            _yolo = YOLO(weights)
                            print(f"[CrowdLens] ✓ Model: {weights}")
                            break
                        except Exception:
                            continue
                    if _yolo is None:
                        raise RuntimeError("No YOLO weights found")
                except Exception as e:
                    print(f"[CrowdLens] YOLO unavailable ({e}) — HOG fallback active")
                    _yolo = "fallback"
    return _yolo

def get_cascade():
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


# ─── NMS ──────────────────────────────────────────────────────────────────────

def nms_boxes(boxes, iou_threshold=0.35):
    if not boxes:
        return []
    arr = np.array([[x1,y1,x2,y2,c] for x1,y1,x2,y2,c in boxes], dtype=np.float32)
    x1s,y1s,x2s,y2s,confs = arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4]
    areas = np.maximum(0,(x2s-x1s)) * np.maximum(0,(y2s-y1s))
    order = confs.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1s[i], x1s[order[1:]])
        yy1 = np.maximum(y1s[i], y1s[order[1:]])
        xx2 = np.minimum(x2s[i], x2s[order[1:]])
        yy2 = np.minimum(y2s[i], y2s[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou   = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[np.where(iou<=iou_threshold)[0]+1]
    return [(int(arr[k,0]),int(arr[k,1]),int(arr[k,2]),int(arr[k,3]),float(arr[k,4])) for k in keep]


# ─── Detection ────────────────────────────────────────────────────────────────

def filter_person_boxes(boxes, frame_h, frame_w):
    """
    Remove detections that are geometrically implausible for a standing person.
    Buildings, signs and vehicles are typically wide (aspect < 1.0) or huge.
    People boxes have: height > width, height ≥ 5 % of frame, reasonable area.
    """
    filtered = []
    min_h = frame_h * 0.05   # must be at least 5 % of frame height
    max_h = frame_h * 0.98   # reject full-frame blobs
    for (x1, y1, x2, y2, conf) in boxes:
        w = x2 - x1;  h = y2 - y1
        if w <= 0 or h <= 0: continue
        aspect = h / w          # > 1 means taller than wide (person-like)
        if aspect < 0.80:       continue   # clearly too wide → building / vehicle
        if aspect > 7.0:        continue   # absurdly tall → artifact
        if h < min_h:           continue   # too small → noise
        if h > max_h:           continue   # full-frame blob → background
        filtered.append((x1, y1, x2, y2, conf))
    return filtered

def detect_people_yolo(frame_bgr, conf=0.35):
    model = get_yolo()
    if model == "fallback":
        return detect_people_hog(frame_bgr)

    H, W = frame_bgr.shape[:2]
    TILE, STEP = 640, int(640 * 0.70)
    all_boxes = []
    xs = sorted(set(list(range(0, max(1,W-TILE), STEP)) + [max(0,W-TILE)]))
    ys = sorted(set(list(range(0, max(1,H-TILE), STEP)) + [max(0,H-TILE)]))

    for y0 in ys:
        for x0 in xs:
            tile = frame_bgr[y0:min(H,y0+TILE), x0:min(W,x0+TILE)]
            # classes=[0] restricts to COCO class 0 = person only
            res  = model(tile, classes=[0], verbose=False, conf=conf)[0]
            for box in res.boxes:
                bx1,by1,bx2,by2 = map(int, box.xyxy[0].tolist())
                all_boxes.append((bx1+x0, by1+y0, bx2+x0, by2+y0, float(box.conf[0])))

    merged = nms_boxes(all_boxes, 0.35)
    return filter_person_boxes(merged, H, W)


def detect_people_hog(frame_bgr):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    H, W = frame_bgr.shape[:2]
    all_boxes = []
    for target in [800, 1200]:
        sc    = min(1.0, target/max(H,W))
        small = cv2.resize(frame_bgr,(int(W*sc),int(H*sc)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rects,wts = hog.detectMultiScale(gray, winStride=(6,6), padding=(8,8), scale=1.03)
        if len(rects):
            for (x,y,bw,bh),wt in zip(rects, np.array(wts).flatten()):
                all_boxes.append((int(x/sc),int(y/sc),int((x+bw)/sc),int((y+bh)/sc),float(wt)))
    return nms_boxes(all_boxes, 0.3)


def detect_extra_heads(frame_bgr, yolo_boxes):
    """Haar cascade supplement — only adds detections clearly missed by YOLO.
    Uses stricter minNeighbors to avoid false positives on buildings/objects."""
    cascade = get_cascade()
    H, W    = frame_bgr.shape[:2]
    gray    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(gray, gray)
    sc      = min(1.0, 960/max(H,W))
    small   = cv2.resize(gray,(int(W*sc),int(H*sc)))
    # minNeighbors=6 and minSize=(35,35) makes it much stricter — only clear human shapes
    rects   = cascade.detectMultiScale(small, scaleFactor=1.05, minNeighbors=6, minSize=(35,35))
    extra   = []
    if not len(rects):
        return extra
    for (x,y,bw,bh) in rects:
        fx1,fy1 = int(x/sc), int(y/sc)
        fx2,fy2 = int((x+bw)/sc), int((y+bh)/sc)
        cx,cy   = (fx1+fx2)//2, (fy1+fy2)//2
        if not any(vx1<=cx<=vx2 and vy1<=cy<=vy2 for (vx1,vy1,vx2,vy2,_) in yolo_boxes):
            extra.append((fx1,fy1,fx2,fy2,0.5))
    return nms_boxes(extra, 0.3)


# ─── Density & visualisation ──────────────────────────────────────────────────

def build_density_map(all_boxes, frame_shape, grid_rows, grid_cols):
    H, W    = frame_shape[:2]
    density = np.zeros((H,W), dtype=np.float32)
    sigma   = max(12, min(H,W)//25)

    for (x1,y1,x2,y2,_) in all_boxes:
        cx,cy = (x1+x2)//2, (y1+y2)//2
        rx0,rx1 = max(0,cx-3*sigma), min(W,cx+3*sigma)
        ry0,ry1 = max(0,cy-3*sigma), min(H,cy+3*sigma)
        xs = np.arange(rx0,rx1); ys = np.arange(ry0,ry1)
        if not (len(xs) and len(ys)): continue
        density[ry0:ry1,rx0:rx1] += np.outer(
            np.exp(-0.5*((ys-cy)/sigma)**2),
            np.exp(-0.5*((xs-cx)/sigma)**2)
        )

    vis     = (density/density.max()*255).astype(np.uint8) if density.max()>0 else density.astype(np.uint8)
    heatmap = cv2.applyColorMap(vis, cv2.COLORMAP_JET)

    cell_counts = np.zeros((grid_rows, grid_cols), dtype=int)
    cell_h, cell_w = H/grid_rows, W/grid_cols
    for (x1,y1,x2,y2,_) in all_boxes:
        cx,cy = (x1+x2)//2,(y1+y2)//2
        cell_counts[min(int(cy/cell_h),grid_rows-1), min(int(cx/cell_w),grid_cols-1)] += 1

    return heatmap, cell_counts, density


def draw_grid_overlay(frame_bgr, cell_counts, boxes, extra_boxes, grid_rows, grid_cols):
    out = frame_bgr.copy()
    H,W = out.shape[:2]
    cell_h, cell_w = H/grid_rows, W/grid_cols
    max_count = max(int(cell_counts.max()),1)

    for r in range(grid_rows):
        for c in range(grid_cols):
            cnt   = int(cell_counts[r,c])
            ratio = cnt/max_count
            color = tuple(int(x) for x in cv2.applyColorMap(
                np.array([[int(ratio*255)]],dtype=np.uint8),cv2.COLORMAP_JET)[0][0])
            x0,y0 = int(c*cell_w),int(r*cell_h)
            x1,y1 = int((c+1)*cell_w),int((r+1)*cell_h)
            ov = out.copy()
            cv2.rectangle(ov,(x0,y0),(x1,y1),color,-1)
            cv2.addWeighted(ov,0.28,out,0.72,0,out)
            cv2.rectangle(out,(x0,y0),(x1,y1),(180,180,180),1)
            if cnt>0:
                cv2.putText(out,str(cnt),(x0+4,y0+16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

    for (x1,y1,x2,y2,conf) in boxes:        # YOLOv8 — green
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,220,90),1)
        cv2.putText(out,f"{conf:.2f}",(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,220,90),1)
    for (x1,y1,x2,y2,_) in extra_boxes:     # Haar cascade — yellow-green
        cv2.rectangle(out,(x1,y1),(x2,y2),(180,255,60),1)
    return out


def frame_to_b64(frame_bgr, quality=88):
    _,buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()

def density_level(pct):
    if pct<20:  return "Low",      "#22c55e"
    if pct<50:  return "Moderate", "#f59e0b"
    if pct<75:  return "High",     "#f97316"
    return              "Critical", "#ef4444"

def compute_density_pct(total, H, W):
    area_units = (H*W)/10000.0
    pct = min(100.0, round(total/max(area_units,1)*100,1))
    return max(pct,1.0) if total>0 else 0.0

def active_model_label():
    m = get_yolo()
    if m == "fallback": return "HOG + SVM (fallback)"
    w = getattr(m,'ckpt_path',None) or "YOLOv8-Large"
    return str(w).split("/")[-1].split("\\")[-1].replace(".pt","").upper()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model_info")
def get_model_info():
    m = get_yolo()
    is_fallback = (m == "fallback")
    return jsonify({
        "pipeline":  "HOG + SVM Pedestrian Detector" if is_fallback else "YOLOv8-Large · Tiled 640 px · 30% overlap",
        "detector":  "Histogram of Oriented Gradients" if is_fallback else "COCO person class · conf ≥ 0.10 · IoU-NMS 0.35",
        "occlusion": "Haar Upper-Body Cascade (OpenCV)" if is_fallback else "Haar Cascade supplement · recovers occluded heads",
        "density":   "Gaussian KDE (σ = frame/25) → JET colormap",
        "binning":   "NumPy spatial cell binning (grid rows × cols)",
        "nms":       "IoU-NMS threshold 0.30" if is_fallback else "IoU-NMS threshold 0.35",
    })


@app.route("/analyze/image", methods=["POST"])
def analyze_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f         = request.files["file"]
        grid_rows = int(request.form.get("grid_rows", 10))
        grid_cols = int(request.form.get("grid_cols", 10))

        # In-memory decode — zero disk I/O for images
        img_bytes = f.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Cannot decode image — ensure PNG or JPG format."}), 400

        H, W      = frame.shape[:2]
        orig_res  = f"{W}×{H}"

        # Preserve aspect ratio; only rescale if above 1920px on longest side
        if max(H,W) > 1920:
            sc    = 1920/max(H,W)
            frame = cv2.resize(frame,(int(W*sc),int(H*sc)), interpolation=cv2.INTER_AREA)
            H,W   = frame.shape[:2]

        t0          = time.time()
        yolo_boxes  = detect_people_yolo(frame)
        extra_boxes = detect_extra_heads(frame, yolo_boxes)
        all_boxes   = yolo_boxes + extra_boxes
        elapsed_ms  = round((time.time()-t0)*1000)

        total                   = len(all_boxes)
        heatmap, cell_counts, _ = build_density_map(all_boxes, frame.shape, grid_rows, grid_cols)
        annotated               = draw_grid_overlay(frame, cell_counts, yolo_boxes, extra_boxes, grid_rows, grid_cols)
        blend                   = cv2.addWeighted(frame,0.52,heatmap,0.48,0)
        density_pct             = compute_density_pct(total, H, W)
        level, level_color      = density_level(density_pct)

        flat     = [(int(cell_counts[r,c]),r,c) for r in range(grid_rows) for c in range(grid_cols)]
        flat.sort(reverse=True)
        hotspots = [{"row":r+1,"col":c+1,"count":cnt} for cnt,r,c in flat[:5] if cnt>0]

        # Row-aggregated series for the line chart (image mode: x = grid row)
        row_series = [{"label":f"Row {r+1}","count":int(sum(cell_counts[r]))} for r in range(grid_rows)]

        return jsonify({
            "total_people":   total,
            "yolo_detected":  len(yolo_boxes),
            "extra_detected": len(extra_boxes),
            "density_pct":    density_pct,
            "level":          level,
            "level_color":    level_color,
            "inference_ms":   elapsed_ms,
            "resolution":     f"{W}×{H}",
            "orig_resolution":orig_res,
            "hotspots":       hotspots,
            "cell_counts":    cell_counts.tolist(),
            "grid_rows":      grid_rows,
            "grid_cols":      grid_cols,
            "row_series":     row_series,
            "original_b64":   frame_to_b64(frame),
            "annotated_b64":  frame_to_b64(annotated),
            "heatmap_b64":    frame_to_b64(blend),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/analyze/video", methods=["POST"])
def analyze_video():
    tmp_path = None
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f           = request.files["file"]
        grid_rows   = int(request.form.get("grid_rows",   10))
        grid_cols   = int(request.form.get("grid_cols",   10))
        sample_rate = int(request.form.get("sample_rate", 30))

        # Save temp file for sequential OpenCV reading
        uid      = uuid.uuid4().hex[:8]
        tmp_path = UPLOAD_FOLDER / f"tmp_{uid}{Path(f.filename).suffix}"
        f.save(str(tmp_path))

        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            return jsonify({"error": "Cannot open video — supported formats: MP4, AVI, MOV."}), 400

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale        = min(1.0, 960/max(H,W,1))
        out_w,out_h  = int(W*scale), int(H*scale)

        acc_density = np.zeros((out_h,out_w), dtype=np.float32)
        timeline    = []
        frame_idx = sampled = 0
        preview   = None
        last_cell_counts = None

        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % sample_rate == 0:
                if scale < 1.0:
                    frame = cv2.resize(frame,(out_w,out_h), interpolation=cv2.INTER_AREA)
                yolo_boxes  = detect_people_yolo(frame)
                extra_boxes = detect_extra_heads(frame, yolo_boxes)
                all_boxes   = yolo_boxes + extra_boxes
                _, cell_counts, density_raw = build_density_map(all_boxes, frame.shape, grid_rows, grid_cols)
                acc_density += density_raw
                last_cell_counts = cell_counts.copy()
                timeline.append({"frame":frame_idx,"time_s":round(frame_idx/fps,2),"count":len(all_boxes)})
                if sampled == 0:
                    preview = (frame.copy(), yolo_boxes, extra_boxes, cell_counts.copy())
                sampled += 1
            frame_idx += 1
            if sampled >= 60: break

        cap.release()
        elapsed = time.time()-t0

        agg_norm = (acc_density/acc_density.max()*255).astype(np.uint8) if acc_density.max()>0 else acc_density.astype(np.uint8)
        agg_heat = cv2.applyColorMap(agg_norm, cv2.COLORMAP_JET)

        counts      = [r["count"] for r in timeline]
        avg_count   = round(sum(counts)/max(len(counts),1),1)
        peak_count  = max(counts) if counts else 0
        density_pct = compute_density_pct(avg_count, out_h, out_w)
        level, level_color = density_level(density_pct)

        prev_b64=ann_b64=heat_b64=""
        final_cell_counts = []
        if preview:
            pf,pb,pe,pc = preview
            blend     = cv2.addWeighted(pf,0.52,agg_heat,0.48,0)
            annotated = draw_grid_overlay(pf,pc,pb,pe,grid_rows,grid_cols)
            prev_b64  = frame_to_b64(pf)
            ann_b64   = frame_to_b64(annotated)
            heat_b64  = frame_to_b64(blend)
            final_cell_counts = pc.tolist()

        return jsonify({
            "total_frames":   total_frames,
            "sampled_frames": sampled,
            "fps":            round(fps,1),
            "duration_s":     round(total_frames/max(fps,1),1),
            "avg_people":     avg_count,
            "peak_people":    peak_count,
            "density_pct":    density_pct,
            "level":          level,
            "level_color":    level_color,
            "inference_s":    round(elapsed,1),
            "resolution":     f"{out_w}×{out_h}",
            "orig_resolution":f"{W}×{H}",
            "timeline":       timeline,
            "original_b64":   prev_b64,
            "annotated_b64":  ann_b64,
            "heatmap_b64":    heat_b64,
            "cell_counts":    final_cell_counts,
            "grid_rows":      grid_rows,
            "grid_cols":      grid_cols,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and Path(tmp_path).exists():
            try: Path(tmp_path).unlink()
            except: pass


@app.route("/analyze/frame", methods=["POST"])
def analyze_frame():
    """
    Real-time single-frame endpoint for live video analysis.
    Accepts a JPEG/PNG frame as multipart 'file' or raw bytes body.
    Returns detection boxes, cell counts, heatmap — all as lightweight JSON.
    Skips Haar cascade by default (too slow for real-time); use ?haar=1 to enable.
    """
    try:
        grid_rows = int(request.form.get("grid_rows", 10))
        grid_cols = int(request.form.get("grid_cols", 10))
        use_haar  = request.form.get("haar", "0") == "1"

        if "file" in request.files:
            img_bytes = request.files["file"].read()
        else:
            img_bytes = request.get_data()

        if not img_bytes:
            return jsonify({"error": "No frame data received"}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Cannot decode frame"}), 400

        # Downscale for speed — keep longest side ≤ 640
        H, W   = frame.shape[:2]
        scale  = min(1.0, 640 / max(H, W, 1))
        if scale < 1.0:
            frame = cv2.resize(frame, (int(W * scale), int(H * scale)),
                               interpolation=cv2.INTER_AREA)
            H, W  = frame.shape[:2]

        t0         = time.time()
        yolo_boxes = detect_people_yolo(frame)
        extra_boxes = detect_extra_heads(frame, yolo_boxes) if use_haar else []
        all_boxes  = yolo_boxes + extra_boxes
        elapsed_ms = round((time.time() - t0) * 1000)

        total = len(all_boxes)
        heatmap, cell_counts, _ = build_density_map(all_boxes, frame.shape, grid_rows, grid_cols)
        blend    = cv2.addWeighted(frame, 0.52, heatmap, 0.48, 0)
        annotated = draw_grid_overlay(frame, cell_counts, yolo_boxes, extra_boxes, grid_rows, grid_cols)

        density_pct        = compute_density_pct(total, H, W)
        level, level_color = density_level(density_pct)

        flat     = [(int(cell_counts[r, c]), r, c)
                    for r in range(grid_rows) for c in range(grid_cols)]
        flat.sort(reverse=True)
        hotspots = [{"row": r + 1, "col": c + 1, "count": cnt}
                    for cnt, r, c in flat[:5] if cnt > 0]

        # Serialize detection boxes for canvas drawing (scaled back to original coords)
        inv = 1.0 / scale if scale > 0 else 1.0
        boxes_out = [{"x1": int(x1*inv), "y1": int(y1*inv),
                      "x2": int(x2*inv), "y2": int(y2*inv),
                      "conf": round(conf, 2), "src": "yolo"}
                     for x1, y1, x2, y2, conf in yolo_boxes]
        boxes_out += [{"x1": int(x1*inv), "y1": int(y1*inv),
                       "x2": int(x2*inv), "y2": int(y2*inv),
                       "conf": round(conf, 2), "src": "haar"}
                      for x1, y1, x2, y2, conf in extra_boxes]

        return jsonify({
            "total_people":  total,
            "density_pct":   density_pct,
            "level":         level,
            "level_color":   level_color,
            "inference_ms":  elapsed_ms,
            "resolution":    f"{W}×{H}",
            "cell_counts":   cell_counts.tolist(),
            "grid_rows":     grid_rows,
            "grid_cols":     grid_cols,
            "hotspots":      hotspots,
            "boxes":         boxes_out,
            "heatmap_b64":   frame_to_b64(blend, quality=75),
            "annotated_b64": frame_to_b64(annotated, quality=75),
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n"+"="*58)
    print("  CrowdLens — Crowd Density Management System")
    print("  http://127.0.0.1:5000")
    print("="*58+"\n")
    get_yolo()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)