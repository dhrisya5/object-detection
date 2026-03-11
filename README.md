# 🎯 Real-Time Object Detection & Multi-Object Tracking

**YOLOv8 + DeepSORT + FastAPI + Streamlit + PostgreSQL/SQLite**

---

## 📁 Project Structure

```
yolo_tracking/
├── main.py              ← Single entry point (runs everything)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── logs/                ← Auto-created: DB, state JSON, log file
└── models/              ← Auto-created: YOLO weights cached here
```

---

## ⚡ Quick Start (Cursor / Local)

### Step 1 — Prerequisites
```bash
# Python 3.9–3.11 required
python --version

# (Optional) create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```
> First run auto-downloads **YOLOv8n weights** (~6 MB) from Ultralytics.

### Step 3 — Run the system

#### ▶ Option A — Demo mode (no webcam needed) ← RECOMMENDED FOR FIRST RUN
```bash
python main.py --mode demo
```
Downloads a short sample video and runs the full pipeline.

#### ▶ Option B — Webcam (default camera)
```bash
python main.py --mode webcam
```

#### ▶ Option C — Webcam (specific index)
```bash
python main.py --mode webcam --source 1
```

#### ▶ Option D — RTSP IP Camera
```bash
python main.py --mode rtsp --source "rtsp://admin:pass@192.168.1.100:554/stream"
```

#### ▶ Option E — API server only
```bash
python main.py --mode api
```

#### ▶ Option F — Dashboard only (while pipeline runs in another terminal)
```bash
python main.py --mode dashboard
```

#### ▶ Option G — All services (demo + API + dashboard)
```bash
python main.py
# or explicitly:
python main.py --mode all
```

---

## 🌐 Access Points

| Service | URL |
|---------|-----|
| **Streamlit Dashboard** | http://localhost:8501 |
| **FastAPI Swagger UI** | http://localhost:8000/docs |
| **Latest Detections** | http://localhost:8000/detections/latest |
| **Stats** | http://localhost:8000/stats |
| **Frame** | http://localhost:8000/frame |
| **History** | http://localhost:8000/detections/history |

---

## 🐳 Docker (optional)

```bash
# SQLite (simplest)
docker build -t tracker .
docker run -p 8000:8000 -p 8501:8501 tracker

# With PostgreSQL
docker-compose up --build
```

---

## 🗄️ PostgreSQL Setup (optional)

By default the system uses **SQLite** — zero configuration needed.

To use PostgreSQL:
```bash
cp .env.example .env
# Edit .env and set DATABASE_URL
```

---

## 🎮 Keyboard Controls (OpenCV window)

| Key | Action |
|-----|--------|
| `Q` | Quit |

---

## 🔧 Config Tuning

Edit the `Config` class in `main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | nano/small/medium/large/xlarge |
| `CONFIDENCE` | `0.40` | Detection confidence threshold |
| `TARGET_CLASSES` | person, car, … | Classes to detect |
| `FRAME_SKIP` | `2` | Process every N-th frame |
| `LINE_Y_FRACTION` | `0.55` | Counting line position |

---

## 🛠 Troubleshooting

**"No module named X"** — Run `pip install -r requirements.txt`

**Black screen / No camera** — Use `--mode demo` for testing

**YOLO download slow** — Pre-download: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"`

**Port in use** — Kill with `lsof -ti:8000 | xargs kill` or `lsof -ti:8501 | xargs kill`
