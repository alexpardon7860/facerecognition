# Face Recognition Attendance System

A modern web-based face recognition attendance system built with Flask and OpenCV.

## Features

- **Real-time Face Recognition**: Live video streaming with face detection and recognition
- **Web Interface**: Modern, responsive web UI with Bootstrap
- **Attendance Tracking**: Automatic attendance marking with CSV export
- **Student Management**: Add new students through web interface
- **Performance Optimized**: Pickle file system for fast face encoding loading
- **Statistics Dashboard**: Real-time attendance statistics and student lists

## Installation

1. **Clone or download the project**
   ```bash
   cd sih
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add student photos**
   - Place student photos in the `face_database/` folder
   - Name format: `StudentName.jpg` (e.g., `John_Doe.jpg`)
   - Ensure photos contain clear, front-facing faces

## Usage

1. **Start the web application**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:5000`
   - The system will automatically process face images on first run

3. **Use the system**
   - Click "Start Camera" to begin attendance tracking
   - Students will be automatically recognized and marked present
   - Use "Add Student" to register new students via webcam
   - View real-time statistics and attendance lists

## File Structure

```
sih/
├── app.py                 # Flask web application
├── facerecognition.py     # Original console-based system
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface template
├── face_database/        # Student photos and encodings
│   ├── *.jpg            # Student photos
│   └── face_encodings.pkl # Cached face encodings
└── attendance_logs/      # Daily attendance CSV files
    └── attendance_YYYY-MM-DD.csv
```

## API Endpoints

- `GET /` - Main web interface
- `GET /video_feed` - Video stream endpoint
- `POST /start_camera` - Start camera for attendance
- `POST /stop_camera` - Stop camera
- `GET /attendance_summary` - Get attendance statistics
- `POST /add_student` - Add new student with photo
- `POST /reset_attendance` - Reset today's attendance

## Deployment

The system is ready for deployment on platforms like:
- Heroku
- Railway
- DigitalOcean
- AWS
- Any platform supporting Python Flask applications

Make sure to:
1. Install system dependencies for OpenCV and dlib
2. Configure camera access permissions
3. Set appropriate environment variables

## Performance Features

- **Pickle Caching**: Face encodings are cached for instant loading
- **Efficient Processing**: Only processes every other frame for better performance
- **Smart Updates**: Adding new students doesn't require full database reload

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

**Note**: Camera access requires HTTPS in production environments.
