# Real-Time Object Detection & Multi-Object Tracking

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


## Access Points

| Service | URL |
|---------|-----|
| **Streamlit Dashboard** | http://localhost:8501 |
| **FastAPI Swagger UI** | http://localhost:8000/docs |
| **Latest Detections** | http://localhost:8000/detections/latest |
| **Stats** | http://localhost:8000/stats |
| **Frame** | http://localhost:8000/frame |
| **History** | http://localhost:8000/detections/history |

---

