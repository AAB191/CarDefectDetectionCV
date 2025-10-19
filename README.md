# Car Damage Detection & Valuation System

A comprehensive AI-powered system for detecting car damage and providing market valuation. The system is split into separate frontend and backend components for easy deployment on Render.

## 🚀 Quick Start

This project is designed for deployment on Render with separate frontend and backend services.

### Backend (FastAPI)
- Located in `/backend` directory
- Provides AI-powered damage detection API
- Fetches real-time market prices
- Calculates damage-adjusted valuations

### Frontend (Static Website)
- Located in `/frontend` directory
- Modern, responsive web interface
- Drag & drop image upload
- Real-time API integration

## 📁 Project Structure

```
├── backend/                 # FastAPI backend service
│   ├── cv/                 # Computer vision modules
│   ├── pricing/            # Market price fetching
│   ├── services/           # Business logic
│   ├── main.py            # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   └── render.yaml        # Render deployment config
├── frontend/               # Static frontend website
│   ├── index.html         # Main application page
│   ├── result.html        # Results template
│   ├── styles.css         # CSS styles
│   ├── config.js          # Configuration
│   └── render.yaml        # Render deployment config
└── README.md              # This file
```

## 🛠️ Features

### AI-Powered Damage Detection
- Detects cracks, dents, scratches, and paint damage
- Severity classification (minor, moderate, severe)
- Visual annotation of damage areas

### Market Valuation
- Real-time price fetching from multiple sources
- Damage-adjusted valuation calculations
- Comparison with similar listings

### Modern UI/UX
- Responsive design for all devices
- Drag & drop file upload
- Real-time form validation
- Loading states and error handling

## 🚀 Deployment on Render

### 1. Deploy Backend

1. Connect your GitHub repository to Render
2. Create a new **Web Service**
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11
   - **Root Directory**: `backend`

### 2. Deploy Frontend

1. Create a new **Static Site**
2. Configure:
   - **Build Command**: `echo "Static site - no build needed"`
   - **Publish Directory**: `.` (root of frontend)
   - **Root Directory**: `frontend`

### 3. Update Configuration

After both services are deployed:

1. Copy your backend URL from Render dashboard
2. Update `frontend/config.js`:
   ```javascript
   API_BASE_URL: 'https://your-backend-service.onrender.com'
   ```
3. Redeploy the frontend

## 🔧 Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
# Serve with any static server
python -m http.server 8001
# or
npx http-server -p 8001
```

Update `frontend/config.js` for local development:
```javascript
API_BASE_URL: 'http://localhost:8000'
```

## 📋 API Endpoints

- `GET /` - Main application page
- `POST /analyze` - Analyze car damage and get valuation
- `GET /health` - Health check

## 🛡️ Security Features

- CORS enabled for cross-origin requests
- File type validation
- File size limits
- Input sanitization

## 📱 Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the README files in `/backend` and `/frontend`
2. Review Render deployment logs
3. Open an issue on GitHub