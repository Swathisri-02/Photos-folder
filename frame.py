import cv2
import os

def split_video_into_frames(video_path, frames_folder):

    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_interval = fps  # Capture every 1 second

    frame_count = 0
    second_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frames_folder, f"frame_{second_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            second_count += 10 # Increment by 5 seconds  

        frame_count += 1

    cap.release()
    print(f"Frames saved in: {frames_folder}")

# # Example usage
# video_path = "sample.mp4"
# frames_folder = "output_frames"
# split_video_into_frames(video_path, frames_folder)