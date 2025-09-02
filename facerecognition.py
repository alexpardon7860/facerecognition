import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime, date
import time
import warnings
import pickle
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
class FaceRecognitionAttendance:
    def __init__(self, face_database_path="face_database", attendance_log_path="attendance_logs"):
        self.face_database_path = face_database_path
        self.attendance_log_path = attendance_log_path
        self.pickle_file = os.path.join(face_database_path, "face_encodings.pkl")
        self.known_face_encodings = []
        self.known_face_names = []
        self.present_students = set()
        self.attendance_marked_today = set()
        
        # Create directories if they don't exist
        os.makedirs(face_database_path, exist_ok=True)
        os.makedirs(attendance_log_path, exist_ok=True)
        
        # Load known faces from pickle or create new
        self.load_known_faces()
        
        # Initialize camera with proper color settings
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        # Ensure color format
        self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        
    def load_known_faces(self):
        """Load face encodings from pickle file or create new from images"""
        # Try to load from pickle file first
        if os.path.exists(self.pickle_file):
            try:
                print("Loading face encodings from pickle file...")
                with open(self.pickle_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"✓ Loaded {len(self.known_face_names)} students from pickle file")
                return
            except Exception as e:
                print(f"Error loading pickle file: {str(e)}")
                print("Rebuilding face database...")
        
        # If pickle doesn't exist or failed to load, process images
        print("Processing face images...")
        self.known_face_encodings = []
        self.known_face_names = []
        
        for filename in os.listdir(self.face_database_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract student name from filename
                student_name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.face_database_path, filename)
                
                try:
                    # Load and encode the face
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)
                    
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(student_name)
                        print(f"✓ Processed {student_name}")
                    else:
                        print(f"✗ No face found in {filename}")
                        
                except Exception as e:
                    print(f"✗ Error processing {filename}: {str(e)}")
        
        # Save to pickle file
        self.save_face_encodings()
        print(f"Loaded {len(self.known_face_names)} students successfully")
    
    def save_face_encodings(self):
        """Save face encodings to pickle file for faster loading"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.pickle_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved face encodings to {self.pickle_file}")
        except Exception as e:
            print(f"Error saving face encodings: {str(e)}")
    
    def mark_attendance(self, name):
        """Mark attendance for a student"""
        current_time = datetime.now()
        today = date.today()
        
        # Check if already marked today
        if name not in self.attendance_marked_today:
            self.attendance_marked_today.add(name)
            self.present_students.add(name)
            
            # Create attendance record
            attendance_record = {
                'Name': name,
                'Date': today.strftime('%Y-%m-%d'),
                'Time': current_time.strftime('%H:%M:%S'),
                'Status': 'Present'
            }
            
            # Save to CSV
            self.save_attendance_record(attendance_record)
            print(f"✓ Attendance marked for {name} at {current_time.strftime('%H:%M:%S')}")
            return True
        return False
    
    def save_attendance_record(self, record):
        """Save attendance record to CSV file"""
        today = date.today()
        filename = f"attendance_{today.strftime('%Y-%m-%d')}.csv"
        filepath = os.path.join(self.attendance_log_path, filename)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filepath)
        
        # Create DataFrame and append to CSV
        df = pd.DataFrame([record])
        df.to_csv(filepath, mode='a', header=not file_exists, index=False)
    
    def run_attendance_system(self):
        """Main loop for the attendance system"""
        print("\n" + "="*50)
        print("FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        print("Press 'q' to quit")
        print("Press 's' to show attendance summary")
        print("="*50 + "\n")
        
        # Variables for processing optimization
        process_this_frame = True
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = self.video_capture.read()
            if not ret:
                print("Failed to capture video")
                break
            
            # Ensure frame is in color (BGR format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Frame is already in color, no conversion needed
                pass
            else:
                # Convert grayscale to color if needed
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Process every other frame to improve performance
            if process_this_frame:
                    # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert BGR to RGB for face_recognition library
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in current frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.6
                    )
                    name = "Unknown"
                    
                    # Use the known face with smallest distance
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding
                    )
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            # Mark attendance
                            self.mark_attendance(name)
                    
                    face_names.append(name)
            
            process_this_frame = not process_this_frame
            
            # Draw rectangles and labels on faces
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Choose color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label background
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Add text
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
            # Add attendance counter
            attendance_text = f"Present Today: {len(self.present_students)}"
            cv2.putText(frame, attendance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Face Recognition Attendance System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.show_attendance_summary()
        
        # Cleanup
        self.video_capture.release()
        cv2.destroyAllWindows()
        
        # Final summary
        self.show_attendance_summary()
    
    def show_attendance_summary(self):
        """Display current attendance summary"""
        print("\n" + "="*40)
        print("ATTENDANCE SUMMARY")
        print("="*40)
        print(f"Date: {date.today().strftime('%Y-%m-%d')}")
        print(f"Total Students in Database: {len(self.known_face_names)}")
        print(f"Present Today: {len(self.present_students)}")
        print(f"Absent: {len(self.known_face_names) - len(self.present_students)}")
        
        if self.present_students:
            print("\nPresent Students:")
            for student in sorted(self.present_students):
                print(f"  ✓ {student}")
        
        absent_students = set(self.known_face_names) - self.present_students
        if absent_students:
            print("\nAbsent Students:")
            for student in sorted(absent_students):
                print(f"  ✗ {student}")
        
        print("="*40 + "\n")
    
    def add_new_student(self, name, image_path):
        """Add a new student to the database"""
        try:
            # Load and verify the image contains a face
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            
            if not face_encodings:
                print(f"No face detected in {image_path}")
                return False
            
            # Copy image to database folder
            import shutil
            destination = os.path.join(self.face_database_path, f"{name}.jpg")
            shutil.copy2(image_path, destination)
            
            # Add to current arrays instead of full reload
            self.known_face_encodings.append(face_encodings[0])
            self.known_face_names.append(name)
            
            # Update pickle file
            self.save_face_encodings()
            
            print(f"✓ Successfully added {name} to the database")
            return True
            
        except Exception as e:
            print(f"Error adding student: {str(e)}")
            return False
    
    def generate_attendance_report(self, start_date=None, end_date=None):
        """Generate attendance report for a date range"""
        if not os.path.exists(self.attendance_log_path):
            print("No attendance records found")
            return
        
        all_records = []
        for filename in os.listdir(self.attendance_log_path):
            if filename.startswith('attendance_') and filename.endswith('.csv'):
                filepath = os.path.join(self.attendance_log_path, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_records.append(df)
                except Exception as e:
                    print(f"Error reading {filename}: {str(e)}")
        
        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            
            # Filter by date range if specified
            if start_date and end_date:
                combined_df['Date'] = pd.to_datetime(combined_df['Date'])
                mask = (combined_df['Date'] >= start_date) & (combined_df['Date'] <= end_date)
                combined_df = combined_df.loc[mask]
            
            # Generate summary
            attendance_summary = combined_df.groupby('Name').size().reset_index(name='Days_Present')
            attendance_summary = attendance_summary.sort_values('Days_Present', ascending=False)
            
            print("\nAttendance Report:")
            print("-" * 30)
            for _, row in attendance_summary.iterrows():
                print(f"{row['Name']:<20} {row['Days_Present']} days")
            
            return attendance_summary
        else:
            print("No attendance records found")
            return None

def main():
    """Main function to run the attendance system"""
    # Initialize the system
    attendance_system = FaceRecognitionAttendance()
    
    if len(attendance_system.known_face_names) == 0:
        print("No students found in database!")
        print("Please add student photos to the 'face_database' folder")
        print("Format: StudentName.jpg (e.g., 'John_Doe.jpg')")
        return
    
    print(f"Loaded {len(attendance_system.known_face_names)} students")
    print("Starting attendance system...")
    
    # Run the main attendance loop
    try:
        attendance_system.run_attendance_system()
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        print("Attendance system shutdown complete")

if __name__ == "__main__":
    main()
