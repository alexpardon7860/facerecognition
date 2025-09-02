# Alternative: Deploy on Render

Since Railway is having issues with system dependencies, try **Render** which handles face recognition libraries better.

## Deploy on Render

1. **Go to [render.com](https://render.com)**
2. **Sign up with GitHub**
3. **Create Web Service**
   - Click "New" → "Web Service"
   - Connect your GitHub repo: `alexpardon7860/facerecognition`

4. **Configure Settings:**
   ```
   Name: face-attendance-system
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn --bind 0.0.0.0:$PORT app:app
   ```

5. **Deploy** - Render automatically handles system dependencies

## Why Render Works Better

- ✅ **Pre-installed libraries**: Has X11 libraries by default
- ✅ **Face recognition friendly**: Better support for OpenCV/dlib
- ✅ **Free tier**: 512MB RAM, enough for face recognition
- ✅ **Automatic HTTPS**: Camera access works properly

## Alternative: Use Dockerfile on Railway

If you prefer Railway, the Dockerfile I created should work:
- Railway will use the Dockerfile automatically
- Installs all required system libraries
- Should resolve the libX11.so.6 error

## Quick Test

Try Render first - it's specifically good for ML/AI applications like face recognition.
