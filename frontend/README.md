# Car Damage Detection & Valuation - Frontend

This is the frontend application for the Car Damage Detection & Valuation system. It's a static website that communicates with the backend API.

## Features

- Modern, responsive UI design
- Drag & drop image upload
- Real-time form validation
- Integration with backend API
- Mobile-friendly interface

## Files

- `index.html` - Main application page with upload form
- `result.html` - Results display template (used by backend)
- `styles.css` - Shared CSS styles
- `render.yaml` - Render deployment configuration

## Local Development

1. Serve the files using any static file server:
```bash
# Using Python
python -m http.server 8001

# Using Node.js (if you have http-server installed)
npx http-server -p 8001

# Using PHP
php -S localhost:8001
```

2. Update the `API_BASE_URL` in `index.html` to point to your backend:
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## Deployment on Render

1. Connect your GitHub repository to Render
2. Create a new Static Site
3. Use the following settings:
   - **Build Command**: `echo "Static site - no build needed"`
   - **Publish Directory**: `.` (root directory)
   - **Environment**: Static

## Configuration

After deploying both frontend and backend, update the `API_BASE_URL` in `index.html` to point to your deployed backend URL:

```javascript
const API_BASE_URL = 'https://your-backend-url.onrender.com';
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+
