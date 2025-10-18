Car Damage Detection & Valuation

Overview
- FastAPI backend with OpenCV damage detection and web scraping for market prices.
- Minimal HTML frontend to upload images and display detection and valuation.

Quickstart
1) Create and activate a virtual environment
```
python -m venv .venv
.venv\\Scripts\\activate
```
2) Install dependencies
```
pip install -r requirements.txt
```
3) Run the server
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
4) Open the app
```
http://localhost:8000
```

Environment
- Copy `.env.example` to `.env` and adjust settings.
 - Optional: enable YOLO model for better accuracy
   - Install: `pip install ultralytics`
   - Download weights to `models/car_damage.pt` (or any YOLOv8/v5 damage model)
   - Set env: `YOLO_MODEL_PATH=./models/car_damage.pt`

Deploy
- Heroku/Railway/Vercel via provided `Dockerfile` or Procfile (see below).

Repository Structure
```
app/
  main.py
  cv/
    __init__.py
    damage.py
  pricing/
    __init__.py
    fetch.py
  services/
    __init__.py
    valuation.py
  schemas.py
  utils.py
static/
  styles.css
templates/
  index.html
  result.html
tests/
  test_damage.py
requirements.txt
Dockerfile
Procfile
README.md
.env.example
.gitignore
```

Notes
- Scraping uses requests+BeautifulSoup first; Selenium is optional and disabled by default.
- For marketplaces with anti-bot protection, consider manual API keys or cached sample data.
 - Detection: Uses YOLO (if weights are present) with OpenCV heuristic as fallback.


