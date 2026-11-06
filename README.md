# Sentinel

**Video Surveillance Analysis System**

Sentinel is an intelligent video surveillance analysis system designed to process CCTV footage using computer vision and AI. It focuses on detecting and tracking objects such as people and vehicles to support surveillance monitoring and video investigation.

## Features

- Object detection using YOLO.
- Multi-object tracking using ByteTrack.
- Video ingestion and metadata extraction.
- High-speed video processing through frame sampling.
- Object trajectory visualization.
- Backend API built with FastAPI.
- PostgreSQL database integration.

## Technology Stack

- **Language:** Python
- **Backend:** FastAPI
- **Computer Vision:** OpenCV
- **Object Detection:** Ultralytics YOLO
- **Object Tracking:** ByteTrack
- **Database:** PostgreSQL
- **Deep Learning:** PyTorch

## Project Setup

### 1. Prerequisites

Install the following:

- Python 3.11
- Anaconda or Miniconda
- PostgreSQL
- FFmpeg and FFprobe
- Git

An NVIDIA GPU is recommended for faster inference but is not mandatory.

### 2. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd sentinel
```

### 3. Create the Environment

```bash
conda create -n sentinel python=3.11
conda activate sentinel
```

### 4. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

Install a compatible PyTorch build for your hardware. For NVIDIA GPU acceleration, follow the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 5. Configure Environment Variables

Create a `.env` file using `.env.example` as a template.

Configure the database connection:

```env
PROJECT_NAME=Sentinel
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/sentinel_db
```

Replace `YOUR_PASSWORD` with your PostgreSQL password.

### 6. Database Setup

Create a PostgreSQL database named `sentinel_db`.

Run database migrations:

```bash
cd backend
alembic upgrade head
```

### 7. Run the Backend

From the `backend` directory:

```bash
uvicorn app.main:app --reload
```

Access the API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Running Detection and Tracking

Activate the Sentinel environment and run the scripts from the project root:

```bash
python scripts/test_detection.py
```

```bash
python scripts/test_tracking.py
```

YOLO model weights are downloaded automatically when required by Ultralytics.

## Project Structure

```text
sentinel/
├── backend/       # FastAPI backend and database
├── frontend/      # Frontend application
├── scripts/       # Detection and tracking scripts
├── models/        # Model configurations and weights
├── storage/       # Video and generated data storage
├── .env.example   # Environment configuration template
├── .gitignore
└── README.md
```

## Notes

- Uploaded videos, generated outputs, model weights, and environment files are excluded from version control.
- Each local setup requires its own database configuration.
- GPU acceleration depends on compatible hardware and drivers.

## License

License and third-party model terms are to be specified. Refer to the respective dependency and model licenses before redistribution or commercial use.
