from flask import Flask, request
import cv2
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import shutil

from frame import split_video_into_frames
from top_ten import process_images
from get_images import get_frames_with_persons

app = Flask(__name__)

# Initialize InsightFace model
face_analyzer = FaceAnalysis(name="buffalo_sc")
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Memory to store known faces
known_faces = {}
next_id = 1  # ID counter for new persons
THRESHOLD = 0.5  # Cosine similarity threshold (lower = stricter)

# Dictionary to store detected persons per image
detected_persons_per_image = {}
persons_to_images = {}
top_10_persons = {}

def recognize_faces(image_path):
    """Detects and assigns unique IDs to persons in a given image."""
    global next_id
    img = cv2.imread(image_path)
    faces = face_analyzer.get(img)

    detected_persons = []  # Stores persons detected in this image

    for face in faces:
        embedding = face.normed_embedding.reshape(1, -1)  # Get embedding

        # Check against known faces
        matched_id = None
        for person_id, known_embedding in known_faces.items():
            similarity = cosine_similarity(embedding, known_embedding)[0][0]
            if similarity > THRESHOLD:  # If similar enough, it's the same person
                matched_id = person_id
                break

        if matched_id is None:
            matched_id = next_id  # Assign new ID
            known_faces[next_id] = embedding  # Store new face
            next_id += 1

        detected_persons.append(matched_id)

        # Draw bounding box and ID
        x, y, w, h = face.bbox.astype(int)
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(img, f"Person_{matched_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, detected_persons


@app.route('/process_video', methods=['POST','GET'])
def all():
    if 'video_path' not in request.files:
        return "No video file provided", 400
    
    video_file = request.files['video_path']  # Get file object
    frames_folder = request.form['frames_folder']
    output_folder = request.form['output_folder']
    result_folder = request.form['result_folder']

    # Save the uploaded file temporarily
    video_path = f"/tmp/{video_file.filename}"  # Adjust path based on your OS
    video_file.save(video_path)  # Save the file

    split_video_into_frames(video_path, frames_folder)

    detected_persons_per_image,top_10_persons = process_images(frames_folder, output_folder)

    # get_frames_with_persons(frames_folder, result_folder, detected_persons_per_image, 2)

    # Generate all possible single and pairwise combinations
    combinations = []
    for r in range(1, 5):  # 1 for single, 2 for pairs
        combinations.extend(itertools.combinations(top_10_persons, r))

    # Process each combination
    for combo in combinations:
        get_frames_with_persons(frames_folder, result_folder, detected_persons_per_image, *combo)

    # Delete frames_folder and output_folder after processing
    try:
        shutil.rmtree(frames_folder)
        print(f"✅ Deleted: {frames_folder}")
    except Exception as e:
        print(f"❌ Error deleting {frames_folder}: {e}")

    try:
        shutil.rmtree(output_folder)
        print(f"✅ Deleted: {output_folder}")
    except Exception as e:
        print(f"❌ Error deleting {output_folder}: {e}")


    return "Processing complete!"

@app.route('/')
def index():
    return "Welcome to the Face Recognition API!"

if __name__ == '__main__':
    app.run(debug=True)
