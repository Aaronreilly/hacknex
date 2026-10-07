🚀 VisionDetect AI

<p align="center">
  <img src="https://img.shields.io/badge/Computer%20Vision-AI-blue?style=for-the-badge" alt="Computer Vision">
  <img src="https://img.shields.io/badge/YOLO11-Custom%20Trained-green?style=for-the-badge" alt="YOLO11">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge" alt="MediaPipe">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge" alt="FastAPI">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge" alt="Python">
  <img src="https://img.shields.io/badge/Status-Hackathon%20Prototype-purple?style=for-the-badge" alt="Hackathon">
</p>

<h1 align="center">VisionDetect AI</h1>

<p align="center">
  <strong>AI-powered person, bag, hand, pose and movement analysis from video.</strong>
</p>

<p align="center">
  <a href="https://hacknex.vercel.app/">🌐 Live Demo</a> •
  <a href="https://github.com/Aaronreilly/hacknex">💻 GitHub Repository</a> •
  <a href="https://github.com/Aaronreilly/hacknex/issues">🐛 Issues</a>
</p>

🌐 Live Application

🔗 Hosted Website

https://hacknex.vercel.app/

The hosted frontend provides the public-facing VisionDetect interface.

Important: The Vercel deployment hosts the web frontend. The custom Python YOLO + MediaPipe video-processing pipeline requires the FastAPI backend described in the local execution section unless the backend is separately deployed and connected to the hosted frontend.

💻 Source Code

https://github.com/Aaronreilly/hacknex

📌 Table of Contents

Overview

Problem

Solution

Features

Architecture

Computer Vision Pipeline

AI Models

Dataset

Event Detection

Project Structure

Technology Stack

Installation

Model Setup

Running Locally

API

Demo

Screenshots

Performance

Limitations

Future Improvements

Team

External Tools and References

Submission Checklist

License

🧠 Overview

VisionDetect AI is a computer-vision system designed to understand interactions between a person and a bag in video.

Instead of treating object detection as the final result, the project combines multiple computer-vision signals:

Person Detection
       +
Bag Detection
       +
Human Pose
       +
Hand Landmarks
       +
Hand-to-Bag Proximity
       +
Movement Over Time
       ↓
Higher-Level Event Detection

The system can identify interaction states such as:

BAG DETECTED
      ↓
PERSON PICKING BAG
      ↓
PERSON PICKED BAG
      ↓
PERSON CARRYING BAG
      ↓
PERSON KEPT BAG
      ↓
BAG PLACED

This architecture demonstrates how several specialized vision models can be combined to produce a more meaningful interpretation of video than a single object detector.

🎯 Problem

Traditional object detection answers questions such as:

"Is there a person?"

"Is there a bag?"

However, for many real-world scenarios, detection alone is not enough.

The system also needs to understand:

Is the person approaching the bag?

Is the person's hand near the bag?

Is the bag moving?

Is the person carrying the bag?

Did the person leave the bag somewhere?

Did the interaction persist over multiple frames?

VisionDetect AI addresses this by combining detection + pose + hand landmarks + spatial relationships + temporal reasoning.

💡 Solution

VisionDetect uses a multi-stage computer-vision pipeline.

Stage 1 — Object Detection

A custom-trained YOLO11n model detects:

person
bag

Stage 2 — Pose Estimation

YOLOv8n-Pose estimates the human body keypoints and draws the skeleton.

Stage 3 — Hand Tracking

MediaPipe Hand Landmarker detects 21 landmarks per hand.

Stage 4 — Interaction Analysis

The system calculates:

Hand-to-bag distance

Object movement

Person movement

Hand state

Temporal persistence

Stage 5 — Event State Machine

The signals are converted into meaningful events.

✨ Features

Feature

Description

🎯 Person Detection

Detects people using a custom YOLO11n model

👜 Bag Detection

Detects bags using the custom-trained detector

✋ Hand Tracking

Detects 21 hand landmarks using MediaPipe

🤏 Gesture Analysis

Estimates OPEN / PARTIAL / GRAB states

🦴 Pose Estimation

Detects body keypoints and skeleton

📏 Proximity Analysis

Measures hand-to-bag distance

🚶 Movement Analysis

Measures frame-to-frame movement

🧠 Event Detection

Converts detections into interaction events

🎥 Video Processing

Generates annotated output video

🌐 Web Interface

Browser-based VisionDetect interface

⚡ REST API

FastAPI backend for AI processing

📊 JSON Results

Returns machine-readable event information

🏗️ Architecture

flowchart TD

    A[👤 User] --> B[🌐 VisionDetect Web UI]

    B -->|Upload Video| C[⚡ FastAPI Backend]

    C --> D[🎯 Custom YOLO11n]
    C --> E[🦴 YOLOv8n-Pose]
    C --> F[✋ MediaPipe Hand Landmarker]

    D --> G[Person + Bag Detections]
    E --> H[Human Pose / Skeleton]
    F --> I[Hand Landmarks + Gesture]

    G --> J[🧠 Interaction Analysis]
    H --> J
    I --> J

    J --> K[📈 Temporal Movement Analysis]

    K --> L[🧠 Event State Machine]

    L --> M[🎥 Annotated Video]
    L --> N[📋 JSON Event Results]

    M --> B
    N --> B

🔬 Computer Vision Pipeline

                  INPUT VIDEO
                       │
                       ▼
               ┌──────────────┐
               │    OpenCV    │
               │ Frame Reader │
               └──────┬───────┘
                      │
       ┌──────────────┼──────────────┐
       │              │              │
       ▼              ▼              ▼
   YOLO11n       YOLOv8n-Pose    MediaPipe
       │              │              │
       ▼              ▼              ▼
 Person + Bag      Skeleton       Hand
 Detection         Keypoints      Landmarks
       │              │              │
       └──────────────┼──────────────┘
                      ▼
             Spatial Relationships
                      │
                      ▼
             Temporal Movement Logic
                      │
                      ▼
              Event State Machine
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Annotated Video      JSON Result

🤖 AI Models

1. 🎯 Custom YOLO11n — Person + Bag Detector

Base model: YOLO11n
Framework: Ultralytics
Task: Object Detection

Classes

Class ID

Class

0

person

1

bag

Trained checkpoint

backend/models/best_person_bag.pt

Validation Results

Class

Precision

Recall

mAP@50

mAP@50-95

person

0.931

0.971

0.984

0.864

bag

0.412

1.000

0.618

0.590

Overall

0.671

0.986

0.801

0.727

Interpretation

The person class performs strongly across the validation set.

The bag class has lower precision because the validation data contained a very small number of bag instances. Therefore, the bag metrics should be treated as an early prototype measurement rather than a definitive production benchmark.

Official resources

Ultralytics YOLO11

Ultralytics Python

YOLO Training

YOLO Prediction

Ultralytics Quickstart

2. 🦴 YOLOv8n-Pose — Human Pose Estimation

Model: YOLOv8n-Pose
Task: Human pose estimation

Model file:

backend/models/yolov8n-pose.pt

Purpose

The pose model is used to:

Estimate human body keypoints

Draw a human skeleton

Provide additional information about body movement

Support the event-analysis pipeline

The pose model is separate from the custom person/bag detector.

Official resource

Ultralytics Documentation

3. ✋ MediaPipe Hand Landmarker

Technology: MediaPipe Tasks Vision
Task: Hand landmark detection

Model:

backend/models/hand_landmarker.task

MediaPipe Hand Landmarker provides 21 landmarks for each detected hand.

The project uses these landmarks to estimate:

OPEN
PARTIAL
GRAB

The hand position is also compared with the detected bag position.

Why hand landmarks matter

A bag being close to a person does not necessarily mean the person has picked it up.

Hand proximity provides an additional signal:

Person
   +
Hand near bag
   +
Bag movement
   +
Temporal persistence
        ↓
Potential pickup interaction

Official resource

MediaPipe Hand Landmarker

🧰 Supporting Technologies

OpenCV

Used for:

Video input/output

Frame processing

Drawing bounding boxes

Drawing skeletons

Drawing hand landmarks

Writing annotated videos

Spatial-distance calculations

OpenCV Documentation

FastAPI

FastAPI provides the backend REST API.

Main endpoint:

POST /process-video

FastAPI Documentation

Uvicorn

Uvicorn runs the FastAPI application.

Uvicorn Documentation

HTML / CSS / JavaScript

Used for the VisionDetect frontend and user interaction.

TensorFlow.js / COCO-SSD

The original frontend implementation used TensorFlow.js with COCO-SSD for browser-based object detection.

For the custom video-analysis pipeline, inference is performed by the Python backend using the custom YOLO model.

📊 Dataset

A custom person + bag object-detection dataset was prepared for this project.

Classes

0 → person
1 → bag

Dataset purpose

The dataset was used to fine-tune YOLO11n for the project's target scenario.

YOLO dataset structure

The expected structure is:

dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml

Dataset considerations

The dataset should only be redistributed if the source images and annotations are legally permitted to be shared.

If the complete dataset is too large for the GitHub repository:

Store it using Git LFS, or

Store it in an appropriate dataset repository/storage service.

Add the dataset access/download instructions to this README.

Do not upload private, restricted or copyrighted material without permission.

🧠 Event Detection

The system uses a state-based approach rather than assuming:

PERSON MOVING = PERSON CARRYING BAG

Instead, multiple signals are combined.

Event sequence

WAITING
   │
   ▼
BAG DETECTED
   │
   ▼
PERSON PICKING BAG
   │
   ▼
PERSON PICKED BAG
   │
   ▼
PERSON CARRYING BAG
   │
   ▼
PERSON KEPT BAG
   │
   ▼
BAG PLACED

Signals used

Object information

Person bounding box
Bag bounding box

Hand information

Hand landmarks
Hand state
Hand-to-bag distance

Movement information

Person movement
Bag movement

Temporal information

The state machine uses persistence across multiple frames to reduce single-frame noise.

📁 Project Structure

hacknex/
│
├── index.html
├── app.js
├── style.css
│
├── backend/
│   ├── main.py
│   ├── person_bag_movement.py
│   │
│   ├── models/
│   │   ├── best_person_bag.pt
│   │   ├── hand_landmarker.task
│   │   └── yolov8n-pose.pt
│   │
│   ├── uploads/
│   └── outputs/
│
├── dataset/
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── docs/
│   └── screenshots/
│
├── .gitignore
├── .gitattributes
└── README.md

💻 Technology Stack

Layer

Technology

Frontend

HTML, CSS, JavaScript

Frontend Hosting

Vercel

Backend

Python

API

FastAPI

Server

Uvicorn

Object Detection

Ultralytics YOLO11n

Custom Model

YOLO11n Person + Bag

Pose

YOLOv8n-Pose

Hand Tracking

MediaPipe Hand Landmarker

Video Processing

OpenCV

Numerical Processing

NumPy

ML Runtime

PyTorch

Version Control

Git + GitHub

Large Files

Git LFS

🌐 Deployment

Frontend — Vercel

The frontend is deployed at:

🔗 https://hacknex.vercel.app/

The Vercel deployment provides the public web interface.

Source repository

https://github.com/Aaronreilly/hacknex

Backend

The current development backend runs locally using:

http://127.0.0.1:8000

The backend can later be deployed to a cloud platform capable of running Python, PyTorch, Ultralytics and MediaPipe.

If the hosted frontend needs to process videos using the custom model, configure its API endpoint to point to the deployed FastAPI backend rather than 127.0.0.1.

⚙️ Installation

Requirements

Python 3.10+

Git

Git LFS for large files

Modern web browser

Windows/Linux/macOS

Sufficient RAM/storage for ML dependencies and models

Clone the repository

git clone https://github.com/Aaronreilly/hacknex.git
cd hacknex

Install Python dependencies

pip install ultralytics torch torchvision opencv-python numpy pandas matplotlib mediapipe fastapi uvicorn python-multipart

If the repository contains requirements.txt:

pip install -r requirements.txt

📦 Model Setup

Required files:

backend/models/
├── best_person_bag.pt
├── hand_landmarker.task
└── yolov8n-pose.pt

Git LFS

For repositories where the large model files are tracked with Git LFS:

git lfs install
git lfs pull

Official resource:

https://git-lfs.com/

Important

Do not commit:

API keys

passwords

.env files containing secrets

private datasets

restricted data

▶️ Running Locally

1. Start the backend

Open PowerShell:

cd "C:\Users\students\Desktop\vision-detect\vision-detect\backend"

Run:

python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000

Backend:

http://127.0.0.1:8000

Health check

Open:

http://127.0.0.1:8000/

Expected response:

{
  "status": "online",
  "message": "VisionDetect AI Backend is running"
}

2. Open API documentation

FastAPI automatically generates Swagger documentation:

http://127.0.0.1:8000/docs

This can be used to test the backend API directly.

3. Start the frontend

Open a second PowerShell:

cd "C:\Users\students\Desktop\vision-detect\vision-detect"

Run:

python -m http.server 5500

Open:

http://127.0.0.1:5500

Local architecture

Browser
   │
   ▼
127.0.0.1:5500
   │
   │ Upload Video
   ▼
127.0.0.1:8000
   │
   ▼
FastAPI
   │
   ├── YOLO11n
   ├── YOLOv8n-Pose
   ├── MediaPipe
   └── OpenCV

🔌 API

Health Check

GET /

Response:

{
  "status": "online",
  "message": "VisionDetect AI Backend is running"
}

Process Video

POST /process-video

Input

Multipart form:

file = video file

Example response

{
  "success": true,
  "frames_processed": 549,
  "final_state": "BAG DETECTED",
  "person_picked": false,
  "person_carried": false,
  "person_kept": false,
  "bag_placed": false,
  "events": [],
  "video_url": "/outputs/example_processed.mp4"
}

🎬 Demo

🌐 Live Demo

Open the deployed application

https://hacknex.vercel.app/

The live application is the primary public-facing demonstration of the project.

🎥 Recommended Hackathon Demo

A 60–120 second demo should show:

1. Open the application

https://hacknex.vercel.app/

2. Upload a test video

Use a video containing:

A visible bag

A person approaching the bag

The person reaching toward the bag

The person lifting/carrying it

The person leaving or placing the bag

3. Show detection

Demonstrate:

PERSON
BAG

4. Show pose

Demonstrate:

Human skeleton

5. Show hand landmarks

Demonstrate:

21 hand landmarks

6. Show event transition

Example:

BAG DETECTED
       ↓
PERSON PICKING BAG
       ↓
PERSON PICKED BAG
       ↓
PERSON CARRYING BAG

7. Show final result

Display:

Processed video

Event state

Detection information

📸 Screenshots

Create:

docs/screenshots/
├── 01-dashboard.png
├── 02-person-bag-detection.png
├── 03-hand-landmarks.png
├── 04-pose-skeleton.png
├── 05-picking-event.png
└── 06-final-result.png

Then add:

## Dashboard

![VisionDetect Dashboard](docs/screenshots/01-dashboard.png)

## Person + Bag Detection

![Person and Bag Detection](docs/screenshots/02-person-bag-detection.png)

## Hand Landmarks

![Hand Landmarks](docs/screenshots/03-hand-landmarks.png)

## Pose Skeleton

![Pose Skeleton](docs/screenshots/04-pose-skeleton.png)

## Event Detection

![Picking Event](docs/screenshots/05-picking-event.png)

## Final Result

![Final Result](docs/screenshots/06-final-result.png)

🎥 Demo Video

Add your final demo video here:

TODO: Add YouTube / Google Drive / GitHub video link

Recommended format:

## Demo Video

[▶️ Watch the VisionDetect AI Demo](YOUR_VIDEO_URL)

📊 Model Performance

Custom Person + Bag Detector

YOLO11n

Overall

Metric

Result

Precision

0.671

Recall

0.986

mAP@50

0.801

mAP@50-95

0.727

Person

Metric

Result

Precision

0.931

Recall

0.971

mAP@50

0.984

mAP@50-95

0.864

Bag

Metric

Result

Precision

0.412

Recall

1.000

mAP@50

0.618

mAP@50-95

0.590

Important evaluation note

The validation set contained only a small number of bag instances.

Therefore:

Person metrics are more reliable.

Bag recall was high.

Bag precision and mAP should be interpreted cautiously.

Increasing the number and diversity of bag examples is a planned improvement.

⚠️ Limitations

Current prototype limitations include:

Bag detection

The bag detector can produce:

missed detections

duplicate boxes

false positives

unstable detections under occlusion

Hand detection

Hands can temporarily disappear when:

the hand is occluded

the hand is too small

the camera angle changes

lighting is poor

Event detection

The event state machine uses heuristics and temporal thresholds.

Therefore, it is not a formally trained action-recognition model.

Environment

Performance depends on:

Camera position

Lighting

Video resolution

Person/bag appearance

Occlusion

Camera movement

Hardware performance

🚀 Future Improvements

Expand the bag dataset

Improve bag detection under occlusion

Add object tracking IDs

Add multi-person tracking

Improve pickup event classification

Improve placement detection

Train a dedicated action-recognition model

Add real-time webcam inference

Add GPU acceleration

Deploy FastAPI backend

Connect hosted frontend to cloud backend

Add event timeline visualization

Add event confidence scores

Add database event history

Add automated evaluation metrics

Add downloadable incident reports

👥 Team

Replace the placeholder names below with the final hackathon team information before submission.

Role

Responsibility

Computer Vision / AI

YOLO training, MediaPipe integration, pose estimation and event logic

Backend

FastAPI API, video processing and model integration

Frontend

VisionDetect interface and result visualization

Dataset / QA

Dataset preparation, testing and validation

UI/UX

Interface design and presentation

🧪 Reproducibility

To reproduce the project:

1. Clone the repository
2. Install Python dependencies
3. Install/initialize Git LFS if required
4. Restore the model files
5. Place models inside backend/models/
6. Start FastAPI
7. Start the frontend server
8. Open the local frontend
9. Upload a test video
10. Review the processed output

🔗 Project Links

Resource

Link

🌐 Live Application

https://hacknex.vercel.app/

💻 GitHub Repository

https://github.com/Aaronreilly/hacknex

⚡ Local Frontend

http://127.0.0.1:5500

🔧 Local Backend

http://127.0.0.1:8000

📚 FastAPI Docs

http://127.0.0.1:8000/docs

Local links work only when the corresponding local servers are running.

📚 External Tools and References

Ultralytics

Used for:

YOLO11n custom object detection

YOLOv8n-Pose

Official documentation:

https://docs.ultralytics.com/

MediaPipe

Used for:

Hand landmark detection

Hand interaction analysis

Official documentation:

https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker

OpenCV

Used for:

Video processing

Frame manipulation

Visualization

Official documentation:

https://docs.opencv.org/

FastAPI

Used for:

REST API

Video upload

Backend integration

Official documentation:

https://fastapi.tiangolo.com/

Uvicorn

Used to run the FastAPI server.

Official documentation:

https://www.uvicorn.org/

Git LFS

Used for large model/dataset files.

Official documentation:

https://git-lfs.com/
