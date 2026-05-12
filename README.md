# KolamKala

KolamKala is a full-stack web application developed as a Final Year BCA Project focused on traditional Indian Kolam pattern analysis and generation using Computer Vision and geometric algorithms.

The application bridges cultural art with modern computational techniques by allowing users to analyze uploaded Kolam images and generate new Kolam designs interactively.

---

# Project Overview

KolamKala was developed to digitally preserve and analyze traditional South Indian Kolam art forms through image processing and rule-based geometric computation.

The system allows users to:

- Upload Kolam images
- Detect symmetry patterns
- Count dots, loops, and line structures
- Calculate geometric complexity
- Classify Kolam types
- Generate Kolam patterns dynamically

This project uses OpenCV-based image analysis and rule-based algorithms instead of machine learning models.

---

# Features

- Kolam image upload and analysis
- Horizontal, vertical, and rotational symmetry detection
- Dot (Pulli) detection
- Loop (Sikku curve) detection
- Line (Kambi stroke) counting
- Complexity score calculation
- Rule-based Kolam classification
- Dynamic Kolam pattern generation
- Interactive browser-based frontend
- REST API support using FastAPI
- OpenAPI/Swagger documentation support

---

# Tech Stack

## Frontend
- HTML5
- CSS3
- Vanilla JavaScript
- Tailwind CSS

## Backend
- Python 3.11
- FastAPI
- Uvicorn

## Computer Vision & Processing
- OpenCV
- NumPy
- Pillow

---

# Project Architecture

```text
KolamKala/
│
├── backend/
│   ├── utils/
│   │   ├── image_utils.py
│   │   └── math_utils.py
│   └── main.py
│
├── lib/
├── screenshots/
├── scripts/
├── documentation/
│
├── README.md
├── package.json
├── pyproject. toml
└── requirements.txt
```
---

## How the Analyzer Works
## Step 1 — Image Preprocessing
Convert image to grayscale
Apply Gaussian blur
Perform adaptive thresholding
## Step 2 — Dot Detection
Detect circular structures using Hough Circle Transform
Count Pulli (dot) structures
## Step 3 — Loop Detection
Apply contour detection
Identify Sikku loop patterns
## Step 4 — Line Detection
Use Probabilistic Hough Line Transform
Detect Kambi stroke structures
## Step 5 — Symmetry Analysis
Detect horizontal symmetry
Detect vertical symmetry
Detect rotational symmetry
## Step 6 — Complexity Scoring
Complexity is calculated using weighted geometric parameters:
1. Dot count
2. Loop count
3. Line count
## Step 7 — Pattern Classification
Patterns are classified into:
1. Pulli
2. Sikku
3. Kambi
4. Mandala
---

## API Endpoints-
## Base URL
http://localhost:8090
## Swagger Documentation
http://localhost:8090/docs
## Generate Kolam
POST /generate
## Analyze Kolam
POST /analyze
## Health Check
GET /health
---

## The project documentation folder contains:

1. Blackbook documentation
2. Research paper
3. Technical documentation
4. Project presentation
5. README PDF version

## Available inside:

/documentation
---

## Future Improvements-
1. AI/ML-based Kolam classification.
2. User authentication system.
3. Cloud deployment.
4. SVG/PNG export support.
5. Multi-image batch analysis.
6. Advanced pattern reconstruction.
7. Dataset creation for ML research.
---

## Limitations-
1. Accuracy depends on image quality.
2. Poor lighting may affect detection.
3. Rule-based classification has limitations.
4. Complex overlapping patterns may reduce accuracy.
5. No database integration currently implemented.
---

## Author

## Palak Kolambe
BCA Student | AI & ML Enthusiast | Python Developer
---

## Final Year Project
Ajeenkya DY Patil University, Pune
---

## Project Guide:
Prof. Sandeep Kulkarni
---


License-

This project is developed for academic and educational purposes.

KolamKala — Bridging traditional Indian art with computational geometry.
