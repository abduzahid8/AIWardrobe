# AliceVision Microservice 🔬

Computer vision microservice for AIWardrobe - provides intelligent keyframe selection, enhanced clothing segmentation, and lighting normalization.

## Quick Start

### Option 1: Docker (Recommended)

```bash
cd alicevision-service
docker-compose up -d
```

The service will be available at `http://localhost:5000`

### Option 2: Local Python

```bash
cd alicevision-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/keyframe` | POST | Select best frame from video |
| `/segment` | POST | Segment clothing from image |
| `/lighting` | POST | Normalize image lighting |
| `/process` | POST | Full pipeline (all steps) |

## Usage Examples

### Keyframe Selection
```bash
curl -X POST http://localhost:5000/keyframe \
  -H "Content-Type: application/json" \
  -d '{"frames": ["base64_frame1...", "base64_frame2..."]}'
```

### Full Pipeline
```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{"frames": ["..."], "normalize_lighting": true}'
```

## Configuration

Set `ALICEVISION_URL` in the main API's `.env`:
```
ALICEVISION_URL=http://localhost:5000
```

## API Documentation

Visit `http://localhost:5000/docs` for interactive Swagger documentation.
