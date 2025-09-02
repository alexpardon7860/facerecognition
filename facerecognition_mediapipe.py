import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime
import pandas as pd

class MediaPipeFaceRecognition:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
    
    def get_face_landmarks(self, image):
        """Extract face landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            # Convert landmarks to numpy array
            face_landmarks = []
            for landmark in landmarks.landmark:
                face_landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(face_landmarks)
        return None
    
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
                    landmarks = self.get_face_landmarks(image)
                    if landmarks is not None:
                        self.known_face_encodings.append(landmarks)
                        self.known_face_names.append(name)
    
    def compare_faces(self, known_encodings, face_encoding, tolerance=0.1):
        """Compare face encodings using cosine similarity"""
        if len(known_encodings) == 0:
            return []
        
        # Calculate cosine similarity
        similarities = []
        for known_encoding in known_encodings:
            # Ensure both arrays have same length
            min_len = min(len(known_encoding), len(face_encoding))
            known_enc = known_encoding[:min_len]
            face_enc = face_encoding[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(known_enc, face_enc)
            norm_a = np.linalg.norm(known_enc)
            norm_b = np.linalg.norm(face_enc)
            
            if norm_a == 0 or norm_b == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_a * norm_b)
            
            similarities.append(similarity > (1 - tolerance))
        
        return similarities
    
    def detect_faces_in_frame(self, frame):
        """Detect faces in a frame and return face locations"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        face_locations = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                
                # Convert to (top, right, bottom, left) format
                top = int(bboxC.ymin * ih)
                left = int(bboxC.xmin * iw)
                bottom = int((bboxC.ymin + bboxC.height) * ih)
                right = int((bboxC.xmin + bboxC.width) * iw)
                
                face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def recognize_faces_in_frame(self, frame):
        """Recognize faces in frame and return names and locations"""
        face_locations = self.detect_faces_in_frame(frame)
        face_names = []
        
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Extract face region
            face_image = frame[top:bottom, left:right]
            
            # Get face encoding
            face_encoding = self.get_face_landmarks(face_image)
            
            if face_encoding is not None:
                # Compare with known faces
                matches = self.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                face_names.append(name)
            else:
                face_names.append("Unknown")
        
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
