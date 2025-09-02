# Deployment Guide

## Quick Deployment Options

### Option 1: Railway (Recommended - Easy & Free)

1. **Create Railway Account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **Deploy from GitHub**
   ```bash
   # Initialize git repository
   git init
   git add .
   git commit -m "Initial commit"
   
   # Push to GitHub (create new repo first)
   git remote add origin https://github.com/yourusername/face-attendance.git
   git push -u origin main
   ```

3. **Connect to Railway**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect Flask and deploy

4. **Environment Variables** (if needed)
   - No special env vars required for basic setup

### Option 2: Render (Free Tier Available)

1. **Create Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Deploy Web Service**
   - Click "New" → "Web Service"
   - Connect GitHub repository
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `python app.py`

### Option 3: Heroku (Requires Credit Card)

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Deploy Commands**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Option 4: Local Network Deployment

1. **Find Your IP Address**
   ```bash
   ipconfig  # Windows
   ifconfig  # Mac/Linux
   ```

2. **Run Application**
   ```bash
   python app.py
   ```

3. **Access from Other Devices**
   - Use `http://YOUR_IP_ADDRESS:5000`
   - Ensure firewall allows port 5000

## Pre-Deployment Checklist

- [x] `Procfile` created
- [x] `requirements.txt` updated
- [x] `runtime.txt` specified
- [x] Production settings configured
- [x] `.gitignore` created

## Important Notes

### Camera Access
- **HTTPS Required**: Most browsers require HTTPS for camera access
- **Local Testing**: Use `http://localhost:5000` for local testing
- **Production**: Deploy to platforms that provide HTTPS

### Face Database
- Upload student photos to `face_database/` folder
- Supported formats: JPG, JPEG, PNG
- Name format: `StudentName.jpg`

### Performance Considerations
- **Memory**: Face recognition requires ~512MB RAM minimum
- **CPU**: Optimized for basic VPS performance
- **Storage**: Minimal storage needed (~50MB)

## Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Ensure HTTPS connection
   - Check browser permissions
   - Try different browsers

2. **Slow Performance**
   - Reduce camera resolution in `app.py`
   - Increase frame skip interval
   - Use fewer face encodings

3. **Memory Issues**
   - Reduce number of students
   - Clear pickle cache and regenerate

### Platform-Specific Notes

**Railway**: 
- Automatic HTTPS
- 500MB RAM free tier
- No credit card required

**Render**:
- Automatic HTTPS
- 512MB RAM free tier
- May sleep after 15min inactivity

**Heroku**:
- Requires credit card for verification
- 512MB RAM
- Sleeps after 30min inactivity

## Security Recommendations

1. **Environment Variables**
   ```python
   # Add to app.py if needed
   SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key')
   ```

2. **CORS Settings** (if needed)
   ```bash
   pip install flask-cors
   ```

3. **Rate Limiting** (optional)
   ```bash
   pip install flask-limiter
   ```

## Monitoring & Maintenance

- Check attendance logs regularly
- Monitor system performance
- Update face database as needed
- Backup attendance CSV files

## Support

For deployment issues:
1. Check platform documentation
2. Verify all files are committed to git
3. Check application logs
4. Ensure camera permissions are granted
