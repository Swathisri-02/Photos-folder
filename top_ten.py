import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize InsightFace model
app = FaceAnalysis(name="buffalo_sc")
app.prepare(ctx_id=0, det_size=(640, 640))

# Memory to store known faces
known_faces = {}
next_id = 1  # ID counter for new persons
THRESHOLD = 0.5  # Cosine similarity threshold (lower = stricter)

# Dictionary to store detected persons per image
detected_persons_per_image = {}
persons_to_images = {}
top_10_persons = {}  # Stores the top 10 persons with the highest frames

def recognize_faces(image_path):
    """Detects and assigns unique IDs to persons in a given image."""
    global next_id
    img = cv2.imread(image_path)
    faces = app.get(img)

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

def process_images(input_folder, output_folder):
    """Processes all images in the input folder and assigns unique person IDs."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            processed_image, detected_persons = recognize_faces(image_path)

            # Store detected persons for this image
            detected_persons_per_image[filename] = detected_persons

            # Save processed image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

    # Map persons to images
    for image, persons in detected_persons_per_image.items():
        for person_id in persons:
            if person_id not in persons_to_images:
                persons_to_images[person_id] = []
            persons_to_images[person_id].append(image)

    # Sort persons by the number of images they appear in
    sorted_persons = sorted(persons_to_images.items(), key=lambda x: len(x[1]), reverse=True)

    # Store only the top 10 persons
    global top_10_persons
    top_10_persons = {person_id: images for person_id, images in sorted_persons[:10]}

    # print("\n🔹 Detected Images:", detected_persons_per_image)
    print("\n🔹 Persons to Images:", persons_to_images)
    # print("\n🏆 Top 10 Persons by Appearance:", top_10_persons)
    print("\nTop 10 Persons:", list(top_10_persons.keys()))

    return detected_persons_per_image, list(top_10_persons.keys())


# # Example usage
# input_folder = "output_frames"
# output_folder = "ten"
# process_images(input_folder, output_folder)
