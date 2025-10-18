from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Optional
import os

# Import your internal modules
from cv.damage import detect_damage
from pricing.fetch import fetch_market_prices
from services.valuation import apply_damage_multiplier
from utils import ensure_dirs, save_image_bytes, bgr_image_to_png_bytes, generate_filename

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")

# Ensure folders exist
ensure_dirs([STATIC_DIR, TEMPLATES_DIR])

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Car Damage Detection & Valuation")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    image: UploadFile = File(...),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    mileage: int = Form(...),
    city: Optional[str] = Form(None),
):
    # Read image bytes
    image_bytes = await image.read()

    # Detect damage
    damage_result = detect_damage(image_bytes)
    annotated_png = bgr_image_to_png_bytes(damage_result.annotated_bgr)

    # Save annotated image
    img_filename = generate_filename(prefix="annotated_", ext=".png")
    img_path = os.path.join(STATIC_DIR, img_filename)
    save_image_bytes(img_path, annotated_png)

    # Fetch market prices
    listings = fetch_market_prices(make=make, model=model, year=year, mileage=mileage, city=city)
    prices = [l.price for l in listings if l.price is not None]
    avg_price = sum(prices) / len(prices) if prices else None
    adjusted_price = apply_damage_multiplier(avg_price, damage_result.severity) if avg_price is not None else None

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "image_url": f"/static/{img_filename}",
            "severity": damage_result.severity,
            "num_regions": damage_result.num_regions,
            "avg_price": avg_price,
            "adjusted_price": adjusted_price,
            "listings": listings,
            "make": make,
            "model": model,
            "year": year,
            "mileage": mileage,
            "city": city,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# -------------------------------
# For local testing
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
