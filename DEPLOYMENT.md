# Deployment Guide for Render

This guide will walk you through deploying the Car Damage Detection & Valuation system on Render.

## Prerequisites

- GitHub repository with the project code
- Render account (free tier available)
- Basic understanding of web services

## Step 1: Prepare Your Repository

1. Ensure your repository has the following structure:
   ```
   ├── backend/
   │   ├── main.py
   │   ├── requirements.txt
   │   ├── render.yaml
   │   └── ... (other backend files)
   ├── frontend/
   │   ├── index.html
   │   ├── config.js
   │   ├── render.yaml
   │   └── ... (other frontend files)
   └── README.md
   ```

2. Commit and push all changes to your GitHub repository.

## Step 2: Deploy Backend Service

1. **Log into Render Dashboard**
   - Go to [render.com](https://render.com)
   - Sign in or create an account

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select your repository

3. **Configure Backend Service**
   - **Name**: `car-damage-backend` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory**: `backend`
   - **Plan**: Free (or your preferred plan)

4. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (5-10 minutes)
   - Note the service URL (e.g., `https://car-damage-backend.onrender.com`)

## Step 3: Deploy Frontend Service

1. **Create New Static Site**
   - Click "New +" → "Static Site"
   - Connect your GitHub repository (same as backend)

2. **Configure Frontend Service**
   - **Name**: `car-damage-frontend` (or your preferred name)
   - **Build Command**: `echo "Static site - no build needed"`
   - **Publish Directory**: `.` (root directory)
   - **Root Directory**: `frontend`
   - **Plan**: Free (or your preferred plan)

3. **Deploy**
   - Click "Create Static Site"
   - Wait for deployment to complete (2-5 minutes)
   - Note the site URL (e.g., `https://car-damage-frontend.onrender.com`)

## Step 4: Connect Frontend to Backend

1. **Update Configuration**
   - Go to your frontend repository
   - Edit `frontend/config.js`
   - Update the `API_BASE_URL` with your backend URL:
     ```javascript
     API_BASE_URL: 'https://your-backend-service.onrender.com'
     ```

2. **Redeploy Frontend**
   - Commit and push the changes
   - Render will automatically redeploy the frontend

## Step 5: Test Your Deployment

1. **Visit Your Frontend**
   - Go to your frontend URL
   - You should see the car damage detection interface

2. **Test the Application**
   - Upload a car image
   - Fill in vehicle details
   - Submit the form
   - Verify that results are displayed

## Troubleshooting

### Backend Issues

**Build Failures:**
- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility
- Review build logs in Render dashboard

**Runtime Errors:**
- Check application logs in Render dashboard
- Verify all import paths are correct
- Ensure environment variables are set

### Frontend Issues

**API Connection Errors:**
- Verify the backend URL in `config.js`
- Check CORS settings in backend
- Ensure backend service is running

**Static Site Issues:**
- Verify all files are in the frontend directory
- Check that `index.html` exists
- Review deployment logs

### Common Solutions

1. **Service Not Starting:**
   - Check start command syntax
   - Verify port configuration
   - Review application logs

2. **Build Timeouts:**
   - Optimize dependencies
   - Use smaller base images
   - Check for infinite loops in build process

3. **Memory Issues:**
   - Upgrade to paid plan if needed
   - Optimize application code
   - Reduce dependency footprint

## Environment Variables

### Backend
- `PORT` - Automatically set by Render
- `PYTHON_VERSION` - Set to 3.11.0

### Frontend
- No environment variables needed for static site

## Monitoring and Maintenance

1. **Monitor Performance**
   - Use Render dashboard to monitor service health
   - Check response times and error rates
   - Monitor resource usage

2. **Update Dependencies**
   - Regularly update Python packages
   - Test updates in development first
   - Deploy updates during low-traffic periods

3. **Backup and Recovery**
   - Keep your GitHub repository updated
   - Document any custom configurations
   - Test disaster recovery procedures

## Scaling Considerations

### Free Tier Limitations
- Services may sleep after inactivity
- Limited build minutes per month
- Shared resources

### Paid Tier Benefits
- Always-on services
- More build minutes
- Dedicated resources
- Custom domains

## Security Best Practices

1. **API Security**
   - Implement rate limiting
   - Add authentication if needed
   - Validate all inputs
   - Use HTTPS only

2. **File Upload Security**
   - Validate file types
   - Limit file sizes
   - Scan for malware
   - Store files securely

## Support Resources

- [Render Documentation](https://render.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GitHub Issues](https://github.com/your-repo/issues)

## Next Steps

After successful deployment:

1. Set up custom domains (if needed)
2. Configure monitoring and alerts
3. Implement CI/CD pipelines
4. Add authentication and user management
5. Scale based on usage patterns
