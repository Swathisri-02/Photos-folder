import os
import cv2

def get_frames_with_persons(image_folder, result_folder, detected_persons_per_image, *person_ids):

    person_ids = set(person_ids)  # Convert input to set for easy comparison
    matching_frames = []

    for frame, persons in detected_persons_per_image.items():
        if person_ids.issubset(set(persons)):  # Check if all given persons exist in this frame
            matching_frames.append(frame)

    if not matching_frames:
        print(f"\n❌ No matching frames found for persons: {person_ids}")
        return []

    # Create subfolder inside result_folder
    folder_name = "_".join(map(str, sorted(person_ids)))  # e.g., "1_6"
    save_folder = os.path.join(result_folder, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n✅ Matching Frames for {person_ids}: {matching_frames}")

    # Save matched frames
    for frame in matching_frames:
        img_path = os.path.join(image_folder, frame)
        img = cv2.imread(img_path)

        if img is not None:
            output_path = os.path.join(save_folder, f"matching_{frame}")
            cv2.imwrite(output_path, img)
            print(f"✅ Image saved: {output_path}")
        else:
            print(f"❌ Image not found: {img_path}")

    return matching_frames

# # List of top 10 persons
# top_10 = [6, 1, 7, 20, 81, 5, 105, 39, 149, 3]

# # Generate all possible single and pairwise combinations
# combinations = []
# for r in range(1, 3):  # 1 for single, 2 for pairs
#     combinations.extend(itertools.combinations(top_10, r))

# # Process each combination
# for combo in combinations:
#     get_frames_with_persons(image_folder, result_folder, detected_persons_per_image, *combo)