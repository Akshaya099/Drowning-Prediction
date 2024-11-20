from flask import Flask, render_template, request, url_for, jsonify
import torch
from PIL import Image
import os
import cv2
import ffmpeg
import logging

# Set up logging for error debugging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)  # Update with your custom model path

# Folder paths
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index(): 
    return render_template('index.html', video_path=PROCESSED_FOLDER)

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process based on file type
    try:
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process Image
            processed_file_path = process_image(file_path)
            file_type = 'image'
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            # Process Video
            processed_file_path = process_video(file_path)
            file_type = 'video'
        else:
            return jsonify({"error": "Unsupported file format"}), 400
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({"error": "An error occurred during file processing"}), 500

    # Get the URL for the processed file
    processed_url = url_for('static', filename='processed/' + os.path.basename(processed_file_path))
    upload_url = url_for('static', filename='uploads/' + os.path.basename(file_path))
    
    return render_template('index.html', uploaded_file=upload_url, processed_file=processed_url, file_type=file_type)

def process_image(file_path):
    """Processes an image and performs object detection."""
    image = Image.open(file_path)
    results = model(image)  # Run detection

    # Render and save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    rendered_image = results.render()[0]
    Image.fromarray(rendered_image).save(processed_image_path)
    return processed_image_path

def process_video(file_path):
    """Processes a video, performs object detection frame by frame, and converts it to a supported format."""
    cap = cv2.VideoCapture(file_path)
    
    if not cap.isOpened(): 
        raise ValueError("Error opening video file") 

    output_file = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on each frame
        results = model(frame)
        processed_frame = results.render()[0]  # Get processed frame
        out.write(processed_frame)  # Write to output video

    cap.release()
    out.release()

    # Convert the processed video to a supported format (e.g., MP4)
    converted_output_file = os.path.splitext(output_file)[0] + '_converted.mp4'
    
    try:
        # Use ffmpeg to convert video
        ffmpeg.input(output_file).output(converted_output_file, vcodec='libx264', acodec='aac').run()
        os.remove(output_file)  # Delete original unconverted video to save space
    except Exception as e:
        logging.error(f"Error during video conversion: {str(e)}")
        raise ValueError("Error during video conversion")

    return converted_output_file

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

