// Configuration for the frontend application
// Update this URL when deploying to point to your backend service

const CONFIG = {
    // For local development
    // API_BASE_URL: 'http://localhost:8000',
    
    // For production - replace with your actual backend URL
    API_BASE_URL: 'https://your-backend-service.onrender.com',
    
    // Timeout for API requests (in milliseconds)
    REQUEST_TIMEOUT: 30000,
    
    // Maximum file size for uploads (in bytes) - 10MB
    MAX_FILE_SIZE: 10 * 1024 * 1024,
    
    // Allowed file types
    ALLOWED_FILE_TYPES: ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
};

// Make config available globally
window.CONFIG = CONFIG;
