import cv2
import os
import sqlite3
from flask import Flask
import pyttsx3
import threading
import mediapipe as mp
from deepface import DeepFace
from flask import jsonify
DATABASE = 'attendance.db'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/faces'
def capture_and_compare_from_saved_image(image_path):
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5)

        mp_drawing = mp.solutions.drawing_utils
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'status': 'error', 'message': 'Failed to load image'})

        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Face Mesh
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return jsonify(
                {'status': 'error', 'message': 'No face detected. Please position your face clearly in the frame.'})

        # Draw face landmarks and connections
        for face_landmarks in results.multi_face_landmarks:
            # Draw all face landmarks
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            # Draw eye contours (left and right eyes)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_LEFT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_RIGHT_EYE,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            # Draw face oval
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec)

            # Estimate shoulders and neck (approximate positions)
            ih, iw, _ = image.shape
            landmarks = face_landmarks.landmark

            # Get chin point (approximate neck position)
            chin = landmarks[152]
            chin_x, chin_y = int(chin.x * iw), int(chin.y * ih)

            # Draw neck line (vertical line down from chin)
            neck_length = int(0.2 * ih)  # 20% of image height
            cv2.line(image, (chin_x, chin_y), (chin_x, chin_y + neck_length), (0, 255, 0), 2)

            # Draw shoulders (horizontal line at neck bottom)
            shoulder_width = int(0.4 * iw)  # 40% of image width
            cv2.line(image,
                     (chin_x - shoulder_width // 2, chin_y + neck_length),
                     (chin_x + shoulder_width // 2, chin_y + neck_length),
                     (0, 255, 0), 2)

        # Save the marked image
        marked_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'marked.jpg')
        cv2.imwrite(marked_image_path, image)

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        cursor.execute("SELECT id, name, photo_path FROM users")
        users = cursor.fetchall()

        for user_id, user_name, photo_path in users:
            if not os.path.isfile(photo_path):
                print(f"⚠️ Skipping {user_name}: photo not found at {photo_path}")
                continue
            try:
                result = DeepFace.verify(
                    img1_path=image_path,
                    img2_path=photo_path,
                    model_name='VGG-Face',
                    enforce_detection=False
                )
                if result['verified']:
                    cursor.execute("INSERT INTO attendance (user_id) VALUES (?)", (user_id,))
                    conn.commit()
                    conn.close()

                    message = f'Welcome {user_name}. Attendance is marked for you.'

                    # ✅ Re-initialize engine freshly
                    # engine = pyttsx3.init()
                    # engine.say(message)
                    # engine.runAndWait()

                    def speak(text):
                        local_engine = pyttsx3.init()
                        local_engine.say(text)
                        local_engine.runAndWait()

                    threading.Thread(target=speak, args=(message,)).start()

                    return jsonify({'status': 'success', 'message': message})
            except Exception as e:
                print(f"Error comparing with {user_name}: {e}")
                continue

        conn.close()
        message = 'Face not recognized, please position your face to the camera'

        def speak(text):
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()

        threading.Thread(target=speak, args=(message,)).start()

        return jsonify({'status': 'error', 'message': message})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
