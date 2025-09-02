from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime, date
import pickle
import json
import base64
from io import BytesIO
from PIL import Image
import threading
import time

app = Flask(__name__)

class WebFaceRecognitionAttendance:
    def __init__(self, face_database_path="face_database", attendance_log_path="attendance_logs"):
        self.face_database_path = face_database_path
        self.attendance_log_path = attendance_log_path
        self.pickle_file = os.path.join(face_database_path, "face_encodings.pkl")
        self.known_face_encodings = []
        self.known_face_names = []
        self.present_students = set()
        self.attendance_marked_today = set()
        self.camera_active = False
        self.video_capture = None
        
        # Performance optimization variables
        self.process_this_frame = True
        self.frame_count = 0
        self.last_recognition_time = 0
        
        # Create directories if they don't exist
        os.makedirs(face_database_path, exist_ok=True)
        os.makedirs(attendance_log_path, exist_ok=True)
        
        # Load known faces
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load face encodings from pickle file or create new from images"""
        if os.path.exists(self.pickle_file):
            try:
                with open(self.pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                return
            except Exception as e:
                print(f"Error loading pickle file: {str(e)}")
        
        # Process images if pickle doesn't exist
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.face_database_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                student_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.face_database_path, filename)
                
                try:
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(student_name)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
        
        self.save_face_encodings()
    
    def save_face_encodings(self):
        """Save face encodings to pickle file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving face encodings: {str(e)}")
    
    def start_camera(self):
        """Initialize camera with optimized settings"""
        if not self.camera_active:
            self.video_capture = cv2.VideoCapture(0)
            # Reduced resolution for better performance
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.video_capture.set(cv2.CAP_PROP_FPS, 15)  # Reduced FPS
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer
            self.camera_active = True
    
    def stop_camera(self):
        """Stop camera"""
        if self.camera_active and self.video_capture:
            self.video_capture.release()
            self.camera_active = False
    
    def generate_frames(self):
        """Generate optimized video frames for streaming"""
        while self.camera_active:
            if self.video_capture is None:
                break
                
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            self.frame_count += 1
            current_time = time.time()
            
            # Process face recognition only every 5th frame and max once per second
            face_locations = []
            face_names = []
            
            if (self.frame_count % 5 == 0 and 
                current_time - self.last_recognition_time > 1.0):
                
                self.last_recognition_time = current_time
                
                # Much smaller frame for processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations:  # Only process encodings if faces found
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, face_encoding, tolerance=0.5
                        )
                        name = "Unknown"
                        
                        if len(self.known_face_encodings) > 0 and True in matches:
                            face_distances = face_recognition.face_distance(
                                self.known_face_encodings, face_encoding
                            )
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index] and face_distances[best_match_index] < 0.5:
                                name = self.known_face_names[best_match_index]
                                self.mark_attendance(name)
                        
                        face_names.append(name)
            
            # Draw rectangles and labels (scale back up)
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 5
                right *= 5
                bottom *= 5
                left *= 5
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 4, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add attendance counter
            attendance_text = f"Present: {len(self.present_students)}"
            cv2.putText(frame, attendance_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Compress frame more aggressively
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def mark_attendance(self, name):
        """Mark attendance for a student with cooldown"""
        if name not in self.attendance_marked_today:
            self.attendance_marked_today.add(name)
            self.present_students.add(name)
            
            # Use threading to avoid blocking video stream
            threading.Thread(target=self._save_attendance_async, args=(name,), daemon=True).start()
            return True
        return False
    
    def _save_attendance_async(self, name):
        """Save attendance record asynchronously"""
        try:
            current_time = datetime.now()
            today = date.today()
            
            attendance_record = {
                'Name': name,
                'Date': today.strftime('%Y-%m-%d'),
                'Time': current_time.strftime('%H:%M:%S'),
                'Status': 'Present'
            }
            
            self.save_attendance_record(attendance_record)
        except Exception as e:
            print(f"Error saving attendance: {e}")
    
    def save_attendance_record(self, record):
        """Save attendance record to CSV file"""
        today = date.today()
        filename = f"attendance_{today.strftime('%Y-%m-%d')}.csv"
        filepath = os.path.join(self.attendance_log_path, filename)
        
        file_exists = os.path.exists(filepath)
        df = pd.DataFrame([record])
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
    
    def get_attendance_summary(self):
        """Get current attendance summary"""
        return {
            'date': date.today().strftime('%Y-%m-%d'),
            'total_students': len(self.known_face_names),
            'present_count': len(self.present_students),
            'absent_count': len(self.known_face_names) - len(self.present_students),
            'present_students': list(self.present_students),
            'absent_students': list(set(self.known_face_names) - self.present_students)
        }
    
    def add_new_student(self, name, image_data):
        """Add new student from base64 image data"""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save image
            image_path = os.path.join(self.face_database_path, f"{name}.jpg")
            image.save(image_path)
            
            # Process face encoding
            face_image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image)
            
            if not face_encodings:
                os.remove(image_path)  # Remove invalid image
                return False, "No face detected in image"
            
            # Add to current arrays
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Update pickle file
            self.save_face_encodings()
            
            return True, f"Successfully added {name} to database"
            
        except Exception as e:
            return False, f"Error adding student: {str(e)}"

# Initialize the attendance system
attendance_system = WebFaceRecognitionAttendance()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(attendance_system.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    attendance_system.start_camera()
    return jsonify({'status': 'success', 'message': 'Camera started'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    attendance_system.stop_camera()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/attendance_summary')
def attendance_summary():
    summary = attendance_system.get_attendance_summary()
    return jsonify(summary)

@app.route('/add_student', methods=['POST'])
def add_student():
    data = request.json
    name = data.get('name')
    image_data = data.get('image')
    
    if not name or not image_data:
        return jsonify({'status': 'error', 'message': 'Name and image are required'})
    
    success, message = attendance_system.add_new_student(name, image_data)
    status = 'success' if success else 'error'
    return jsonify({'status': status, 'message': message})

@app.route('/reset_attendance', methods=['POST'])
def reset_attendance():
    attendance_system.present_students.clear()
    attendance_system.attendance_marked_today.clear()
    return jsonify({'status': 'success', 'message': 'Attendance reset for today'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
