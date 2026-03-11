"""
======================================================================
  Real-Time Object Detection & Multi-Object Tracking â€” main.py
  Run:  python main.py [--mode webcam|rtsp|demo|api|dashboard|all]
======================================================================
"""

import os, sys, argparse, threading, time, logging, json, math
import warnings; warnings.filterwarnings("ignore")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 0.  Dependency bootstrap (install missing packages silently)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
REQUIRED = [
    "ultralytics", "opencv-python", "numpy", "fastapi", "uvicorn[standard]",
    "sqlalchemy", "psycopg2-binary", "streamlit", "deep_sort_realtime",
    "pydantic", "python-dotenv", "httpx", "Pillow", "requests",
]

def _bootstrap():
    import importlib, subprocess
    mapping = {
        "opencv-python": "cv2",
        "uvicorn[standard]": "uvicorn",
        "psycopg2-binary": "psycopg2",
        "deep_sort_realtime": "deep_sort_realtime",
        "python-dotenv": "dotenv",
        "Pillow": "PIL",
    }
    for pkg in REQUIRED:
        mod = mapping.get(pkg, pkg.split("[")[0].replace("-", "_"))
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[bootstrap] Installing {pkg} â€¦")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

_bootstrap()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1.  Standard imports (after bootstrap)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Any

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2.  Logging
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s â€” %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)
log = logging.getLogger("main")
Path("logs").mkdir(exist_ok=True)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3.  Configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class Config:
    # Detection
    YOLO_MODEL        = "yolov8n.pt"          # nano â€“ auto-downloaded
    CONFIDENCE        = 0.40
    IOU_THRESHOLD     = 0.45
    TARGET_CLASSES    = ["person", "car", "motorcycle", "bus", "truck", "bicycle"]
    FRAME_SKIP        = 2                      # process every N-th frame
    RESIZE_WIDTH      = 640

    # Counting line  (y-fraction of frame height)
    LINE_Y_FRACTION   = 0.55

    # Database (SQLite fallback if PostgreSQL unavailable)
    DB_URL            = os.getenv("DATABASE_URL", "sqlite:///logs/detections.db")

    # API
    API_HOST          = "0.0.0.0"
    API_PORT          = 8000

    # Shared state directory
    STATE_FILE        = "logs/state.json"

cfg = Config()

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4.  Shared in-memory state  (thread-safe via lock)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import threading as _thr

_state_lock = _thr.Lock()
_state: Dict[str, Any] = {
    "detections": [],          # last frame detections
    "counts": defaultdict(int),
    "crossed": defaultdict(int),
    "fps": 0.0,
    "frame_count": 0,
    "last_frame_b64": "",
    "running": False,
}

def state_get(key):
    with _state_lock:
        return _state[key]

def state_set(key, value):
    with _state_lock:
        _state[key] = value

def state_update_counts(label):
    with _state_lock:
        _state["counts"][label] += 1

def state_update_crossed(label):
    with _state_lock:
        _state["crossed"][label] += 1

def _dump_state():
    """Persist state to JSON so Streamlit dashboard can read it."""
    with _state_lock:
        snap = {
            "counts":  dict(_state["counts"]),
            "crossed": dict(_state["crossed"]),
            "fps":     _state["fps"],
            "frame_count": _state["frame_count"],
            "detections": _state["detections"][-50:],
            "last_frame_b64": _state["last_frame_b64"],
        }
    try:
        Path(cfg.STATE_FILE).write_text(json.dumps(snap))
    except Exception:
        pass

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5.  Database layer
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class DetectionLog(Base):
    __tablename__ = "detection_logs"
    id         = Column(Integer, primary_key=True, index=True)
    timestamp  = Column(DateTime, default=datetime.utcnow)
    object_id  = Column(Integer)
    label      = Column(String(64))
    confidence = Column(Float)
    bbox_x1    = Column(Float); bbox_y1 = Column(Float)
    bbox_x2    = Column(Float); bbox_y2 = Column(Float)
    crossed    = Column(Integer, default=0)
    frame_no   = Column(Integer)

class CrossingEvent(Base):
    __tablename__ = "crossing_events"
    id        = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    label     = Column(String(64))
    object_id = Column(Integer)

def init_db():
    engine = create_engine(cfg.DB_URL, connect_args={"check_same_thread": False}
                           if "sqlite" in cfg.DB_URL else {})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    log.info(f"Database ready: {cfg.DB_URL}")
    return engine, SessionLocal

_engine, _SessionLocal = init_db()

def db_log_detection(obj_id, label, conf, bbox, crossed, frame_no):
    try:
        sess = _SessionLocal()
        rec = DetectionLog(
            object_id=obj_id, label=label, confidence=round(conf, 3),
            bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3],
            crossed=int(crossed), frame_no=frame_no,
        )
        sess.add(rec); sess.commit(); sess.close()
    except Exception as e:
        log.debug(f"db_log_detection error: {e}")

def db_log_crossing(label, obj_id):
    try:
        sess = _SessionLocal()
        sess.add(CrossingEvent(label=label, object_id=obj_id))
        sess.commit(); sess.close()
    except Exception as e:
        log.debug(f"db_log_crossing error: {e}")

def db_recent(n=100):
    try:
        sess = _SessionLocal()
        rows = sess.query(DetectionLog).order_by(
            DetectionLog.id.desc()).limit(n).all()
        sess.close()
        return [
            {"id": r.id, "timestamp": str(r.timestamp),
             "object_id": r.object_id, "label": r.label,
             "confidence": r.confidence, "crossed": r.crossed}
            for r in rows
        ]
    except Exception as e:
        log.debug(f"db_recent error: {e}")
        return []

def db_stats():
    try:
        sess = _SessionLocal()
        total = sess.query(DetectionLog).count()
        crossed = sess.query(CrossingEvent).count()
        sess.close()
        return {"total_detections": total, "total_crossings": crossed}
    except Exception as e:
        return {"total_detections": 0, "total_crossings": 0}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6.  YOLOv8 Detector
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class YOLODetector:
    def __init__(self):
        from ultralytics import YOLO
        log.info(f"Loading YOLO model: {cfg.YOLO_MODEL}")
        self.model = YOLO(cfg.YOLO_MODEL)
        self.names = self.model.names          # {0: 'person', ...}
        self.target_ids = {
            k for k, v in self.names.items()
            if v.lower() in [c.lower() for c in cfg.TARGET_CLASSES]
        }
        log.info(f"Target classes: {[self.names[i] for i in self.target_ids]}")

    def detect(self, frame: np.ndarray):
        """Returns list of (bbox_xyxy, conf, class_id, label)."""
        results = self.model.predict(
            frame, conf=cfg.CONFIDENCE, iou=cfg.IOU_THRESHOLD,
            classes=list(self.target_ids), verbose=False,
        )[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            out.append(([x1, y1, x2, y2], conf, cls, self.names[cls]))
        return out

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7.  DeepSORT Tracker
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class DeepSORTTracker:
    def __init__(self):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.7,
            max_cosine_distance=0.3,
            nn_budget=100,
            embedder="mobilenet",
            half=False,
            bgr=True,
        )
        log.info("DeepSORT tracker initialised")

    def update(self, detections, frame):
        """
        detections: list of ([x1,y1,x2,y2], conf, cls, label)
        Returns: list of (track_id, bbox_xyxy, label, conf)
        """
        ds_input = []
        meta     = []
        for bbox, conf, cls, label in detections:
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], conf, label))
            meta.append((label, conf))

        tracks = self.tracker.update_tracks(ds_input, frame=frame)
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid  = track.track_id
            ltrb = track.to_ltrb()
            label = track.get_det_class() or "object"
            conf  = track.get_det_conf() or 0.0
            results.append((tid, [int(v) for v in ltrb], label, conf))
        return results

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8.  Line Counter
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class LineCounter:
    """Counts objects that cross a horizontal line."""

    def __init__(self, line_y: int):
        self.line_y   = line_y
        self._prev_cy: Dict[int, float] = {}
        self._counted: set = set()

    def update(self, tracks):
        """tracks: list of (tid, bbox_xyxy, label, conf)
           Returns set of (tid, label) that crossed THIS frame."""
        new_crossings = []
        for tid, bbox, label, conf in tracks:
            cy = (bbox[1] + bbox[3]) / 2.0
            prev = self._prev_cy.get(tid)
            if prev is not None and tid not in self._counted:
                if (prev < self.line_y <= cy) or (cy <= self.line_y < prev):
                    self._counted.add(tid)
                    new_crossings.append((tid, label))
            self._prev_cy[tid] = cy
        return new_crossings

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9.  Drawing helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
PALETTE = [
    (255, 56,  56), (255, 157, 151), (255, 112,  31),
    (255, 178, 29), (207, 210,  49), ( 72, 249,  10),
    (146, 204,  23), ( 61, 219, 134), ( 26, 147,  52),
    ( 0, 212, 187), ( 44, 153, 168), ( 0, 194, 255),
    ( 52,  69, 147), (100,  45, 255), (111, 187, 255),
    (255,  84, 232),
]

def _color(tid):
    return PALETTE[int(str(tid).strip()) % len(PALETTE)]

def draw_tracks(frame, tracks, line_y, crossed_ids):
    # Counting line
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
    cv2.putText(frame, "COUNTING LINE", (10, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for tid, bbox, label, conf in tracks:
        x1, y1, x2, y2 = bbox
        col = _color(tid)
        is_crossed = tid in crossed_ids
        thickness = 3 if is_crossed else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thickness)
        txt = f"ID:{tid} {label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), col, -1)
        cv2.putText(frame, txt, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame

def draw_hud(frame, fps, counts, crossed):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 30 + 20 * (len(counts) + 2)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, f"FPS: {fps:.1f}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y = 45
    for label, cnt in sorted(counts.items()):
        cross_cnt = crossed.get(label, 0)
        cv2.putText(frame, f"{label}: {cnt} (x:{cross_cnt})", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
        y += 20
    return frame

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 10. Video pipeline
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
import base64

class VideoPipeline:
    def __init__(self, source=0):
        self.source   = source
        self.detector = YOLODetector()
        self.tracker  = DeepSORTTracker()
        self._crossed_ids: set = set()

    def _open_capture(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")
        return cap

    def run(self, display=True, headless=False):
        cap = self._open_capture()
        ret, sample = cap.read()
        if not ret:
            raise RuntimeError("Cannot read first frame from source.")

        h, w = sample.shape[:2]
        new_w = cfg.RESIZE_WIDTH
        new_h = int(h * new_w / w)
        line_y = int(new_h * cfg.LINE_Y_FRACTION)
        counter = LineCounter(line_y)

        frame_idx = 0
        fps_clock = time.time()
        fps_buffer = []

        state_set("running", True)
        log.info(f"Pipeline started | source={self.source} | display={display}")

        # Per-frame label counts reset each frame
        frame_counts: Dict[str, int] = defaultdict(int)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Stream ended or frame read failed.")
                    break

                frame_idx += 1
                frame = cv2.resize(frame, (new_w, new_h))

                # FPS
                now = time.time()
                fps_buffer.append(now)
                fps_buffer = [t for t in fps_buffer if now - t < 1.0]
                fps = len(fps_buffer)

                # Skip frames for speed
                if frame_idx % cfg.FRAME_SKIP != 0:
                    continue

                # â”€â”€ Detection
                detections = self.detector.detect(frame)

                # â”€â”€ Tracking
                tracks = self.tracker.update(detections, frame)

                # â”€â”€ Line counting
                new_crossings = counter.update(tracks)
                for tid, label in new_crossings:
                    self._crossed_ids.add(tid)
                    state_update_crossed(label)
                    db_log_crossing(label, tid)
                    log.info(f"Crossing: {label} ID={tid}")

                # â”€â”€ Update shared state
                frame_counts.clear()
                det_list = []
                for tid, bbox, label, conf in tracks:
                    state_update_counts(label)
                    frame_counts[label] += 1
                    db_log_detection(tid, label, conf, bbox,
                                     tid in self._crossed_ids, frame_idx)
                    det_list.append({
                        "track_id": tid, "label": label,
                        "conf": round(conf, 3), "bbox": bbox,
                        "crossed": tid in self._crossed_ids,
                    })

                state_set("detections", det_list)
                state_set("fps", fps)
                state_set("frame_count", frame_idx)

                # â”€â”€ Draw
                vis = draw_tracks(frame.copy(), tracks, line_y, self._crossed_ids)
                with _state_lock:
                    counts_snap  = dict(_state["counts"])
                    crossed_snap = dict(_state["crossed"])
                vis = draw_hud(vis, fps, counts_snap, crossed_snap)

                # Encode frame for API/dashboard
                _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode()
                state_set("last_frame_b64", b64)
                _dump_state()

                if display and not headless:
                    cv2.imshow("Object Detection & Tracking", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("User pressed Q â€” stopping.")
                        break

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt â€” stopping pipeline.")
        finally:
            state_set("running", False)
            cap.release()
            if display and not headless:
                cv2.destroyAllWindows()
            log.info("Pipeline stopped.")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 11. FastAPI application
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Object Detection & Tracking API",
    description="Real-time YOLOv8 + DeepSORT detection API",
    version="1.0.0",
)

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/detections/latest")
def latest_detections():
    return {
        "fps": state_get("fps"),
        "frame_count": state_get("frame_count"),
        "detections": state_get("detections"),
        "counts": dict(state_get("counts")),
        "crossed": dict(state_get("crossed")),
    }

@app.get("/detections/history")
def detection_history(limit: int = 100):
    return db_recent(limit)

@app.get("/stats")
def stats():
    return {
        **db_stats(),
        "live_counts":   dict(state_get("counts")),
        "live_crossings": dict(state_get("crossed")),
        "fps":           state_get("fps"),
    }

@app.get("/frame")
def get_frame():
    b64 = state_get("last_frame_b64")
    if not b64:
        raise HTTPException(status_code=404, detail="No frame available yet")
    return {"frame_base64": b64}

def start_api(host=None, port=None):
    import uvicorn
    uvicorn.run(app, host=host or cfg.API_HOST, port=port or cfg.API_PORT,
                log_level="warning")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 12. Streamlit dashboard (written to a temp file and launched)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DASHBOARD_CODE = '''
import streamlit as st, json, time, base64, os, requests
from pathlib import Path
from collections import defaultdict

STATE_FILE = "logs/state.json"
API_URL    = "http://localhost:8000"

st.set_page_config(page_title="Object Tracker Dashboard",
                   page_icon="ðŸŽ¯", layout="wide")

st.title("ðŸŽ¯ Real-Time Object Detection & Tracking Dashboard")
st.caption("YOLOv8 + DeepSORT | FastAPI backend")

def load_state():
    try:
        return json.loads(Path(STATE_FILE).read_text())
    except Exception:
        return {}

def try_api(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=2)
        return r.json()
    except Exception:
        return {}

# Auto-refresh
refresh = st.sidebar.slider("Auto-refresh (sec)", 1, 10, 2)
st.sidebar.markdown("---")
st.sidebar.markdown("**API Endpoints**")
st.sidebar.code(f"{API_URL}/health\\n{API_URL}/stats\\n{API_URL}/detections/latest")

placeholder = st.empty()

while True:
    state = load_state()
    counts  = state.get("counts", {})
    crossed = state.get("crossed", {})
    fps     = state.get("fps", 0)
    fc      = state.get("frame_count", 0)
    dets    = state.get("detections", [])
    b64     = state.get("last_frame_b64", "")

    with placeholder.container():
        # â”€â”€ Top metrics
        cols = st.columns(4)
        cols[0].metric("FPS", f"{fps:.1f}")
        cols[1].metric("Frames Processed", fc)
        cols[2].metric("Active Objects", len(dets))
        cols[3].metric("Total Crossings", sum(crossed.values()))

        st.markdown("---")
        c1, c2 = st.columns([2, 1])

        with c1:
            st.subheader("ðŸ“¹ Live Stream")
            if b64:
                img_bytes = base64.b64decode(b64)
                st.image(img_bytes, channels="BGR",
                         caption="Latest Detection Frame", use_column_width=True)
            else:
                st.info("Waiting for video streamâ€¦ Run: python main.py --mode webcam")

        with c2:
            st.subheader("ðŸ“Š Object Counts")
            if counts:
                import pandas as pd
                df = pd.DataFrame(
                    [{"Object": k, "Detected": counts.get(k,0),
                      "Crossed Line": crossed.get(k,0)}
                     for k in counts],
                ).sort_values("Detected", ascending=False)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.subheader("ðŸš¦ Line Crossings")
                if crossed:
                    st.bar_chart(crossed)
                else:
                    st.caption("No crossings yet.")
            else:
                st.info("No detections yet.")

        st.markdown("---")
        st.subheader("ðŸ” Current Detections")
        if dets:
            import pandas as pd
            df2 = pd.DataFrame(dets)[["track_id","label","conf","crossed"]]
            df2.columns = ["Track ID","Label","Confidence","Crossed Line"]
            st.dataframe(df2, use_container_width=True, hide_index=True)

        # DB stats from API
        api_stats = try_api("/stats")
        if api_stats:
            st.markdown("---")
            st.subheader("ðŸ—ƒï¸ Database Stats")
            c3, c4 = st.columns(2)
            c3.metric("Total DB Detections", api_stats.get("total_detections", 0))
            c4.metric("Total DB Crossings",  api_stats.get("total_crossings",  0))

    time.sleep(refresh)
'''

def start_dashboard():
    import subprocess
    dash_path = Path("logs/dashboard_app.py")
    dash_path.write_text(DASHBOARD_CODE, encoding="utf-8")
    log.info("Starting Streamlit dashboard on http://localhost:8501")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(dash_path),
         "--server.port", "8501", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 13. Demo mode (no webcam required)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_demo():
    """Generates synthetic frames and runs the full pipeline."""
    log.info("Demo mode — using synthetic video stream.")
    source = _make_synthetic_source()
    while True:
        try:
            pipeline = VideoPipeline(source=source)
            pipeline.run(display=True, headless=False)
        except KeyboardInterrupt:
            break

def _make_synthetic_source():
    """Write a tiny synthetic video and return its path."""
    path = "logs/synthetic.avi"
    out  = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"XVID"), 20, (640, 480))
    for i in range(300):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Moving rectangle to simulate object
        x = int((i * 3) % 580)
        cv2.rectangle(frame, (x, 200), (x + 60, 280), (0, 200, 100), -1)
        cv2.putText(frame, "DEMO", (x + 5, 245),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        out.write(frame)
    out.release()
    return path

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 14. Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    parser = argparse.ArgumentParser(description="Object Detection & Tracking System")
    parser.add_argument(
        "--mode",
        choices=["webcam", "rtsp", "demo", "api", "dashboard", "all"],
        default="all",
        help=(
            "webcam   â€“ live webcam (requires camera)\n"
            "rtsp     â€“ RTSP stream (set --source)\n"
            "demo     â€“ sample/synthetic video (no camera needed)\n"
            "api      â€“ FastAPI server only\n"
            "dashboardâ€“ Streamlit dashboard only\n"
            "all      â€“ demo + api + dashboard (DEFAULT)\n"
        ),
    )
    parser.add_argument("--source", default=None,
                        help="Webcam index (0,1â€¦) or RTSP URL")
    parser.add_argument("--headless", action="store_true",
                        help="Suppress OpenCV display window")
    args = parser.parse_args()

    print("\n" + "â•"*60)
    print("  ðŸŽ¯  Real-Time Object Detection & Tracking System")
    print("  Model   : YOLOv8 (auto-download on first run)")
    print("  Tracker : DeepSORT")
    print("  DB      : SQLite (logs/detections.db)")
    print("  API     : http://localhost:8000")
    print("  Dashboard: http://localhost:8501")
    print("â•"*60 + "\n")

    if args.mode == "api":
        start_api()

    elif args.mode == "dashboard":
        start_dashboard()
        log.info("Dashboard launched. Open http://localhost:8501")
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            pass

    elif args.mode == "webcam":
        source = int(args.source) if args.source and args.source.isdigit() \
                 else (args.source or 0)
        # Start API + dashboard in background
        threading.Thread(target=start_api, daemon=True).start()
        time.sleep(1)
        start_dashboard()
        pipeline = VideoPipeline(source=source)
        pipeline.run(display=not args.headless, headless=args.headless)

    elif args.mode == "rtsp":
        if not args.source:
            log.error("--source required for RTSP mode. Example: --source rtsp://...")
            sys.exit(1)
        threading.Thread(target=start_api, daemon=True).start()
        time.sleep(1)
        start_dashboard()
        pipeline = VideoPipeline(source=args.source)
        pipeline.run(display=not args.headless, headless=args.headless)

    elif args.mode in ("demo", "all"):
        # Start API server in background thread
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        time.sleep(1.5)

        # Start Streamlit dashboard
        start_dashboard()
        time.sleep(1)

        log.info("Open http://localhost:8501 for the dashboard")
        log.info("Open http://localhost:8000/docs for the API")
        log.info("Press Q in the video window (or Ctrl+C) to stop.")

        # Run demo pipeline in main thread
        run_demo()

if __name__ == "__main__":
    main()
    


