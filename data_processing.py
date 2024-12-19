import os
import json
import shutil
import numpy as np
from bagpy import bagreader
import pandas as pd
import yaml

def process_bag_to_npy(rosbag_path, output_npy_path):
    """
    Process a single rosbag file and save its content as .npy.
    """
    # Get the directory containing the output file
    output_dir = os.path.dirname(output_npy_path)

    # Check if the output folder exists and contains more than 10 files
    if os.path.exists(output_dir):
        num_files = len([f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])
        if num_files > 100:  # You can adjust this threshold as needed
            print(f"[INFO] Step already done: {output_dir} contains more than 45 files. Skipping.")
            return  # Skip processing

    try:
        # Use bagpy to extract the /USER/markers topic
        b = bagreader(rosbag_path)
        predic_msg = b.message_by_topic(topic='/USER/markers')

        # Read the extracted CSV file directly using predic_msg
        df_predic = pd.read_csv(predic_msg).replace(", ", "\n", regex=True)

        # Initialize an empty list to store frames
        f = []
        for row in range(len(df_predic)):
            this_f = df_predic["markers"][row][1:-1]
            yaml_blocks = this_f.split('head')
            yaml_blocks = yaml_blocks[1:]
            data = [yaml.safe_load(block) for block in yaml_blocks if block.strip()]
            if this_f is None:
                continue
            one_f = np.zeros((17, 3))
            for i in range(17):
                one_f[data[i]["id"] - 500, 0] = data[i]["pose"]["position"]["x"]
                one_f[data[i]["id"] - 500, 1] = data[i]["pose"]["position"]["y"]
                one_f[data[i]["id"] - 500, 2] = data[i]["pose"]["position"]["z"]
            f.append(one_f)

        # Convert list of frames to numpy array and reshape it
        f = np.array(f).reshape(len(f), 17, 3, 1)
        f = np.transpose(f, (1, 0, 2, 3))  # Format: (17, frames, 3, 1)

        # Save the numpy array to the output file
        np.save(output_npy_path, f)
        print(f"[SUCCESS] Processed and saved: {output_npy_path}")

    except Exception as e:
        print(f"[ERROR] Error processing {rosbag_path}: {e}")

    finally:
        # Delete the folder created by bagreader
        if os.path.exists(b.datafolder):
            shutil.rmtree(b.datafolder)
            print(f"[INFO] Deleted temporary folder: {b.datafolder}")

def normalize_data(f):
    """
    Normalize the data by centering and scaling.
    """
    # Calculate torso center for x, y, z
    center_x = (f[5, :, 0, 0] + f[6, :, 0, 0] + f[11, :, 0, 0] + f[12, :, 0, 0]) / 4
    center_y = (f[5, :, 1, 0] + f[6, :, 1, 0] + f[11, :, 1, 0] + f[12, :, 1, 0]) / 4
    center_z = (f[5, :, 2, 0] + f[6, :, 2, 0] + f[11, :, 2, 0] + f[12, :, 2, 0]) / 4

    # Compute std
    std = pow(f[5, :, 0, 0] - center_x, 2) + pow(f[5, :, 1, 0] - center_y, 2) + pow(f[5, :, 2, 0] - center_z, 2)
    std += pow(f[6, :, 0, 0] - center_x, 2) + pow(f[6, :, 1, 0] - center_y, 2) + pow(f[6, :, 2, 0] - center_z, 2)
    std += pow(f[11, :, 0, 0] - center_x, 2) + pow(f[11, :, 1, 0] - center_y, 2) + pow(f[11, :, 2, 0] - center_z, 2)
    std += pow(f[12, :, 0, 0] - center_x, 2) + pow(f[12, :, 1, 0] - center_y, 2) + pow(f[12, :, 2, 0] - center_z, 2)

    std = pow(std/4, 0.5)

    # Avoid division by zero for torso_size
    std[std == 0] = 1

    # Center keypoints
    f[:, :, 0, 0] = (f[:, :, 0, 0] - center_x) / std
    f[:, :, 1, 0] = (f[:, :, 1, 0] - center_y) / std
    f[:, :, 2, 0] = (f[:, :, 2, 0] - center_z) / std

    return f

def generate_json_for_file(f, gesture, output_dir, video_idx, category_id, frames_length):
    """
    Generate and save JSON files for each gesture based on sliding window.
    Returns the updated video_idx to ensure unique JSON naming.
    """
    i = 0  # Window index within the video
    num_frames = f.shape[1]
    while True:
        start_idx = i + 50
        end_idx = start_idx + frames_length
        if end_idx > num_frames:
            break

        tmp = f[:, start_idx:end_idx, :, :]

        # Prepare JSON data
        data = {
            "info": {
                "video_name": f"{gesture}_video_{video_idx}.mp4",
                "resolution": [f.shape[2], f.shape[1]],
                "num_frame": end_idx - start_idx,
                "num_keypoints": f.shape[0],
                "keypoint_channels": ["x", "y", "z"],
                "version": "1.0"
            },
            "annotations": [],
            "category_id": category_id
        }

        for frame_index in range(tmp.shape[1]):
            keypoints = []
            for k in range(tmp.shape[0]):
                x = tmp[k, frame_index, 0, 0]
                y = tmp[k, frame_index, 1, 0]
                z = tmp[k, frame_index, 2, 0]
                keypoints.append([x, y, z])

            annotation = {
                "frame_index": frame_index,
                "id": frame_index,
                "person_id": None,
                "keypoints": keypoints
            }
            data["annotations"].append(annotation)

        # Save JSON
        json_file_path = os.path.join(output_dir, f'{video_idx}.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"[INFO] Saved {json_file_path}")
        i += 32
        video_idx += 1  # Increment global video index for unique naming

    return video_idx


def process_files_by_gesture(Gesture_names, npy_output_folder, frames_length):
    # Prepare gesture mapping
    bag_by_class = {}
    for gesture in Gesture_names:
        gesture_files = [f for f in os.listdir(npy_output_folder) if f.startswith(gesture) and f.endswith('.npy')]
        if len(gesture_files) == 0:
            print(f"[WARNING] No files found for gesture: {gesture}")
        bag_by_class[gesture] = gesture_files

    gesture_to_category = {gesture: idx for idx, gesture in enumerate(Gesture_names)}

    global_video_idx = 0  # Unique JSON file naming across all videos

    set_names = ['train', 'val', 'test']

    for gesture, files in bag_by_class.items():
        if len(files) == 0:
            print(f"[INFO] Skipping gesture '{gesture}' with no files.")
            continue  # Skip gestures with no files

        category_id = gesture_to_category[gesture]

        for name in files:
            print(f"Processing {name} for gesture {gesture}")
            file_path = os.path.join(npy_output_folder, name)

            try:
                f = np.load(file_path)  # Load .npy file
            except Exception as e:
                print(f"[ERROR] Unable to load file {name}: {e}")
                continue

            # Check shape
            if f.ndim != 4 or f.shape[2] != 3:
                print(f"[ERROR] File {name} has unexpected shape {f.shape}. Skipping.")
                continue

            num_frames = f.shape[1]
            if num_frames == 0:
                print(f"[WARNING] File {name} has no frames. Skipping.")
                continue

            print(f"[INFO] File shape: {f.shape}")

            # Normalize Data
            try:
                f = normalize_data(f)
            except Exception as e:
                print(f"[ERROR] Normalization failed for {name}: {e}")
                continue

            # Split the data into three parts
            train_length = int(num_frames * 0.7)
            val_length = int(num_frames * 0.15)
            test_length = num_frames - train_length - val_length

            parts = [
                f[:, 0:train_length, :, :],
                f[:, train_length:train_length + val_length, :, :],
                f[:, train_length + val_length:, :, :]
            ]

            # Assign each part to train, val, and test sets respectively
            for idx, part in enumerate(parts):
                set_name = set_names[idx % 3]
                output_dir = os.path.join(f'./data/assistive_furniture/frame_{frames_length}/{gesture}', set_name)
                os.makedirs(output_dir, exist_ok=True)

                # Generate JSON files
                try:
                    global_video_idx = generate_json_for_file(part, gesture, output_dir, global_video_idx, category_id,
                                                              frames_length)
                except Exception as e:
                    print(f"[ERROR] Failed to generate JSON for {name} in {set_name} set: {e}")

def copy_and_rename_json_files(src_directory, dst_directory, set_names, gestures):
    """
    Copy and rename JSON files from source gesture/set folders into a single destination folder per set.
    """
    for set_name in set_names:
        set_dst_dir = os.path.join(dst_directory, f'kinetics_{set_name}')
        os.makedirs(set_dst_dir, exist_ok=True)
        i = 0  # File index within the set
        for gesture in gestures:
            gesture_set_dir = os.path.join(src_directory, gesture, set_name)
            if not os.path.exists(gesture_set_dir):
                continue
            for file in os.listdir(gesture_set_dir):
                if file.endswith('.json'):
                    src_file_path = os.path.join(gesture_set_dir, file)
                    new_file_name = f'{i}.json'
                    dst_file_path = os.path.join(set_dst_dir, new_file_name)
                    try:
                        shutil.copy2(src_file_path, dst_file_path)
                        i += 1
                    except Exception as e:
                        print(f"Could not copy file {src_file_path} to {dst_file_path} due to {e}")
        print(f"Copied {i} files to {set_dst_dir}")

def rename_files(directory):
    """
    Rename JSON files in a directory to ensure sequential numbering.
    """
    i = 0
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.json'):
            old_file_path = os.path.join(directory, filename)
            new_filename = f"{i}.json"
            new_file_path = os.path.join(directory, new_filename)
            os.rename(old_file_path, new_file_path)
            i += 1
            print(f'Renamed file: {filename} to {new_filename}')

def label_generator(directory_path, if_test):
    label = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                label_id = data["category_id"] if "category_id" in data else None
                if label_id is not None:
                    label.append({'file_name': file_name, 'label_index': label_id})
                else:
                    print(f"File {file_name} does not contain 'category_id' field.")
    output_name = f"{directory_path}_label.json"
    with open(output_name, 'w') as json_file:
        json.dump(label, json_file, indent=4)



if __name__ == "__main__":
    rosbag_folder = "./rosbag"  # Path to the folder containing rosbag videos
    npy_output_folder = "./npy_output"

    # Process all rosbags in the folder
    for rosbag_file in os.listdir(rosbag_folder):
        if rosbag_file.endswith(".bag"):
            rosbag_path = os.path.join(rosbag_folder, rosbag_file)
            npy_output_path = os.path.join(npy_output_folder, rosbag_file.replace(".bag", ".npy"))
            print(f"Processing: {rosbag_file}")
            process_bag_to_npy(rosbag_path, npy_output_path)

    # Define hyperparameters for the pipeline
    Gesture_names = ["Waving_come", "Waving_leave", "Pointing", "Stop", "ODD"]  # List of gesture names
    frames_length = 64  # Input frame length
    rosbag_folder = "./rosbag"  # Path to the folder containing rosbag videos
    npy_output_folder = "./npy_output"  # Folder to save processed .npy files
    os.makedirs(npy_output_folder, exist_ok=True)


    process_files_by_gesture(Gesture_names, npy_output_folder, frames_length)

    # Set the frame length and data folder structure
    src_folder = f"./data/assistive_furniture/frame_{frames_length}"
    dst_folder = f"./data/assistive_furniture/frame_{frames_length}"

    # Define set names
    set_names = ['train', 'val', 'test']

    # Combine JSON files into kinetics_train, kinetics_val, and kinetics_test folders
    copy_and_rename_json_files(src_folder, dst_folder, set_names, Gesture_names)

    # Rename JSON files in each folder
    rename_files(f'{dst_folder}/kinetics_train')
    rename_files(f'{dst_folder}/kinetics_val')
    rename_files(f'{dst_folder}/kinetics_test')

    # Generate label files for each set
    label_generator(f'{dst_folder}/kinetics_train', True)
    label_generator(f'{dst_folder}/kinetics_val', True)
    label_generator(f'{dst_folder}/kinetics_test', True)