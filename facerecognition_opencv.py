import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd

class OpenCVFaceRecognition:
    def __init__(self):
        # Load OpenCV's pre-trained face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Simple face recognition using template matching
        self.known_faces = {}
        self.known_face_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from face_database directory"""
        face_database_path = "face_database"
        if not os.path.exists(face_database_path):
            return
        
        for filename in os.listdir(face_database_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract name from filename
                name = os.path.splitext(filename)[0]
                name = name.replace('_', ' ').replace('-', ' ')
                
                # Load and process image
                image_path = os.path.join(face_database_path, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    
                    # Detect faces in the image
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    if len(faces) > 0:
                        # Take the first face found
                        (x, y, w, h) = faces[0]
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Resize to standard size for comparison
                        face_roi = cv2.resize(face_roi, (100, 100))
                        
                        self.known_faces[name] = face_roi
                        self.known_face_names.append(name)
    
    def compare_faces(self, face_roi, threshold=0.6):
        """Compare face using template matching"""
        if len(self.known_faces) == 0:
            return "Unknown"
        
        # Resize face to standard size
        face_roi = cv2.resize(face_roi, (100, 100))
        
        best_match = "Unknown"
        best_score = 0
        
        for name, known_face in self.known_faces.items():
            # Use template matching
            result = cv2.matchTemplate(face_roi, known_face, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_score and max_val > threshold:
                best_score = max_val
                best_match = name
        
        return best_match
    
    def detect_faces_in_frame(self, frame):
        """Detect faces in a frame and return face locations"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Convert to (top, right, bottom, left) format for consistency
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
        
        return face_locations
    
    def recognize_faces_in_frame(self, frame):
        """Recognize faces in frame and return names and locations"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_locations = []
        face_names = []
        
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize face
            name = self.compare_faces(face_roi)
            
            # Convert to (top, right, bottom, left) format
            face_locations.append((y, x + w, y + h, x))
            face_names.append(name)
        
        return face_locations, face_names
    
    def mark_attendance(self, name):
        """Mark attendance for recognized person"""
        if name == "Unknown":
            return False
        
        # Create attendance_logs directory if it doesn't exist
        os.makedirs("attendance_logs", exist_ok=True)
        
        # Get current date
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Check if already marked today
        attendance_file = f"attendance_logs/attendance_{date_str}.csv"
        
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            if name in df['Name'].values:
                return False  # Already marked
        else:
            df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
        
        # Add new attendance record
        new_record = pd.DataFrame({
            'Name': [name],
            'Time': [time_str],
            'Date': [date_str]
        })
        
        df = pd.concat([df, new_record], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        
        return True
    
    def get_face_landmarks(self, image):
        """Simple face detection for compatibility"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the first face as a simple encoding
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            return face_roi.flatten()  # Return flattened array as "encoding"
        
        return None
