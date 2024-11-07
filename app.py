import cv2
import mediapipe as mp
from flask import Flask, Response

# Initialize Flask app
app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Drawing specifications
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))

# List of landmarks to outline
OUTLINE_LANDMARKS = [
    63, 68, 54, 103, 67, 
    109, 10, 338, 297, 284, 298, 
    293, 334, 296, 336, 
    9, 107, 66, 105
]

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face mesh detection
        result = face_mesh.process(rgb_frame)

        # If face landmarks are detected
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, _ = frame.shape

                # Store landmark coordinates for the outline
                outline_points = []
                for idx in OUTLINE_LANDMARKS:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    outline_points.append((x, y))
                
                # Draw the outline
                for i in range(len(outline_points)):
                    start_point = outline_points[i]
                    end_point = outline_points[(i + 1) % len(outline_points)]  # Connect to the next point
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)  # Draw lines between points
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the output frame in the format required by Flask
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
        <html>
            <head>
                <title>Face Landmark Detection</title>
            </head>
            <body>
                <h1>Real-time Face Landmark Detection</h1>
                <img src="/video_feed" style="width: 100%; max-width: 800px;" />
            </body>
        </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


