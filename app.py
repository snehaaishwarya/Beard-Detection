from flask import Flask, Response, render_template, request, send_file, jsonify
import cv2
import boto3
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)  # 0 is the default camera

# Path where the captured image will be saved
CAPTURED_IMAGE_PATH = 'captured_image.jpg'

# Initialize AWS Rekognition client
rekognition_client = boto3.client('rekognition', region_name='us-east-1')

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    success, frame = camera.read()
    if success:
        cv2.imwrite(CAPTURED_IMAGE_PATH, frame)
        return send_file(CAPTURED_IMAGE_PATH, mimetype='image/jpeg')
    return "Failed to capture image", 500

@app.route('/analyze', methods=['POST'])
def analyze():
    with open(CAPTURED_IMAGE_PATH, 'rb') as image_file:
        response = rekognition_client.detect_faces(
            Image={'Bytes': image_file.read()},
            Attributes=['ALL']  # Request all attributes
        )
    
    # Extract beard and mustache details from the response
    faces = response.get('FaceDetails', [])
    results = []
    for face in faces:
        beard = face.get('Beard', {})
        mustache = face.get('Mustache', {})

        # Handle case where attributes might be boolean values
        beard_value = beard.get('Value', 'Not available')
        beard_confidence = beard.get('Confidence', 'Not available') if isinstance(beard, dict) else 'Not available'
        mustache_value = mustache.get('Value', 'Not available')
        mustache_confidence = mustache.get('Confidence', 'Not available') if isinstance(mustache, dict) else 'Not available'

        results.append({
            'Beard': f"{beard_value} (Confidence: {beard_confidence}%)",
            'Mustache': f"{mustache_value} (Confidence: {mustache_confidence}%)"
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
