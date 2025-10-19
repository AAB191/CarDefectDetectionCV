# Car Damage Detection & Valuation - Backend API

This is the backend API for the Car Damage Detection & Valuation system built with FastAPI.

## Features

- AI-powered car damage detection using computer vision
- Real-time market price fetching from multiple sources
- Damage severity assessment and valuation adjustment
- RESTful API endpoints for frontend integration

## API Endpoints

- `GET /` - Serves the main application page
- `POST /analyze` - Analyzes car damage and returns valuation
- `GET /health` - Health check endpoint

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Deployment on Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

## Environment Variables

- `PORT` - Port number (automatically set by Render)
- `PYTHON_VERSION` - Python version (3.11.0)

## Dependencies

- FastAPI - Web framework
- OpenCV - Computer vision processing
- NumPy - Numerical computations
- Pillow - Image processing
- Requests - HTTP client for price fetching
- BeautifulSoup4 - Web scraping
- Jinja2 - Template engine
