# Deploy Face Recognition App on Render

Railway is having persistent issues with face recognition libraries. **Render.com is the recommended solution** for ML/AI applications.

## Quick Deploy Steps

### 1. Go to Render.com
- Visit [render.com](https://render.com)
- Sign up with your GitHub account

### 2. Create Web Service
- Click **"New"** → **"Web Service"**
- Connect repository: `alexpardon7860/facerecognition`
- Branch: `main`

### 3. Configure Settings
```
Name: face-attendance-system
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn --bind 0.0.0.0:$PORT app:app
```

### 4. Deploy
- Click **"Create Web Service"**
- Render will automatically build and deploy
- Takes 5-10 minutes for first deployment

## Why Render Works

✅ **Pre-installed X11 libraries** - No libX11.so.6 errors  
✅ **ML/AI optimized** - Built for face recognition apps  
✅ **Free tier** - 512MB RAM, sufficient for your app  
✅ **Automatic HTTPS** - Camera access works properly  
✅ **Zero configuration** - No Dockerfile needed  

## Expected Result

Your app will be available at: `https://face-attendance-system.onrender.com`

The deployment should succeed without any library errors.

## Backup: Heroku Alternative

If Render doesn't work, try Heroku with the same configuration:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn --bind 0.0.0.0:$PORT app:app`
