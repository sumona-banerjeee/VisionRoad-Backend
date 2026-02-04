# VisionRoad Backend

AI-powered road infrastructure detection system for analyzing dashcam videos. Detects potholes and signboards with GPS-based location mapping.

## Features

- 🎥 **Video Upload & Processing** - Upload dashcam footage with GPS data
- 🕳️ **Pothole Detection** - YOLO-based pothole detection with tracking
- 🪧 **Signboard Detection** - Traffic sign recognition and classification
- 📍 **GPS Mapping** - Auto-map detections to hierarchical locations
- 🔄 **Real-time Updates** - WebSocket-based progress notifications
- 📊 **Analytics API** - Summary endpoints by project/package/location

## Tech Stack

- **Framework**: FastAPI
- **ML**: PyTorch, Ultralytics YOLO
- **Database**: SQLite + SQLAlchemy + Alembic
- **Real-time**: WebSockets

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/VisionRoad-Backend.git
cd VisionRoad-Backend
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Database Migrations

```bash
# Generate migration (first time setup)
alembic revision --autogenerate -m "initial setup"

# Apply migrations
alembic upgrade head
```

### 5. Start Server

```bash
python main.py
```

Server runs at: **http://localhost:8000**

---

## Project Structure

```
VisionRoad-Backend/
├── app/
│   ├── __init__.py          # FastAPI app initialization
│   ├── core/
│   │   ├── config.py        # Configuration settings
│   │   └── storage.py       # In-memory storage
│   ├── db/
│   │   ├── database.py      # Database connection
│   │   ├── crud.py          # CRUD operations
│   │   └── crud_hierarchy.py # Project/Package/Location CRUD
│   ├── models/
│   │   ├── video.py         # Video model
│   │   ├── detection.py     # Detection model
│   │   ├── project.py       # Project model
│   │   ├── package.py       # Package model
│   │   ├── location.py      # Location model
│   │   └── processing.py    # Processing status model
│   ├── routes/
│   │   ├── upload_process_routes.py  # Upload & processing
│   │   ├── project_routes.py         # Project management
│   │   ├── package_routes.py         # Package management
│   │   ├── location_routes.py        # Location management
│   │   └── summary_routes.py         # Analytics endpoints
│   ├── services/
│   │   ├── upload_service.py      # Upload handling
│   │   ├── video_processor.py     # Pothole detection
│   │   ├── signboard_detector.py  # Signboard detection
│   │   └── location_mapper.py     # GPS-to-location mapping
│   └── ws/
│       └── websocket_manager.py   # WebSocket handling
├── models/                   # YOLO model weights (.pt files)
├── alembic/                  # Database migrations
├── uploads/                  # Uploaded videos (gitignored)
├── results/                  # Detection results (gitignored)
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/upload` | POST | Upload video + GPS JSON |
| `/api/v1/status/{video_id}` | GET | Get processing status |
| `/api/v1/results/{video_id}` | GET | Get detection results |
| `/ws/{video_id}` | WS | Real-time progress updates |
| `/api/v1/projects` | GET/POST | Project management |
| `/api/v1/packages` | GET/POST | Package management |
| `/api/v1/locations` | GET/POST | Location management |
| `/api/v1/summary/projects/{id}` | GET | Project detection summary |

---

## Upload Format

### Video Upload Request

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@dashcam_video.mp4" \
  -F "json_file=@gps_data.json" \
  -F "detection_type=pothole-detection" \
  -F "speed_kmh=30"
```

### GPS JSON Format

```json
{
  "videoFilename": "dashcam_video.mp4",
  "gpsPoints": [
    { "lat": 22.75, "lng": 88.15, "timestamp": 0 },
    { "lat": 22.76, "lng": 88.14, "timestamp": 1 }
  ],
  "totalGpsPoints": 2,
  "durationSeconds": 2
}
```

### Detection Types

- `pothole-detection` - Detect road potholes
- `sign-board-detection` - Detect traffic signboards

---

## Hierarchical Location System

Organize surveys in a hierarchy:

```
Project (e.g., "Kolkata-Jaipur Highway")
  └── Package (e.g., "Kolkata-Patna Section")
        └── Location (e.g., "KM 0-80 Segment")
              └── Video → Detections
```

### Create Hierarchy

```bash
# 1. Create Project
curl -X POST http://localhost:8000/api/v1/projects \
  -H "Content-Type: application/json" \
  -d '{"name": "Highway Survey", "state": "West Bengal", "corridor_name": "NH-19"}'

# 2. Create Package
curl -X POST http://localhost:8000/api/v1/packages \
  -H "Content-Type: application/json" \
  -d '{"project_id": "<PROJECT_ID>", "name": "Section A", "region": "Kolkata"}'

# 3. Create Location (with GPS bounds)
curl -X POST http://localhost:8000/api/v1/locations \
  -H "Content-Type: application/json" \
  -d '{
    "package_id": "<PACKAGE_ID>",
    "segment_name": "KM 0-50",
    "start_lat": 22.5, "start_lng": 88.3,
    "end_lat": 23.0, "end_lng": 87.8,
    "chainage_start_km": 0, "chainage_end_km": 50
  }'
```

Detections with GPS coordinates within a location's bounds are automatically mapped!

---

## Seed Sample Data

```bash
python seed_data.py
```

Creates sample projects for Indian highways (NH-19, NH-44, NH-48).

---

## Environment Variables

Create `.env` file (optional):

```env
DATABASE_URL=sqlite:///./visionroad.db
UPLOAD_DIR=uploads
RESULTS_DIR=results
```

---

## GPU Support

The system auto-detects CUDA. For GPU acceleration:

1. Install CUDA Toolkit 11.8+
2. Install PyTorch with CUDA:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

Check GPU status in logs:
```
Model loaded on GPU: NVIDIA GeForce RTX 3080
```

---

## License

MIT License

---

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
