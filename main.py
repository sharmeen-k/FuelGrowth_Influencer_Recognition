import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
import csv
from urllib.request import urlopen
from multiprocessing import Pool, Manager, cpu_count

# Directory to store face dataset
res = np.zeros((300, 300)) # first index = face number, second = performance
DATASET_DIR = "face_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

def download_video(url: str) -> str:
    # Download video from URL and save locally for temporary usage 
    try:
        print("Downloading video...")
        temp_path = f"temp_{os.path.basename(url).split('?')[0]}"
        with urlopen(url) as response:
            with open(temp_path, "wb") as f:
                f.write(response.read())
        print("Finished downloading")
        return temp_path
    except Exception as e:
        print(f"Failed to download video {url}: {e}")
        return None

def process_video(url: str, performance, face_id_counter, lock):
    # Download video, process frames, and save face encodings.
    video_path = download_video(url)
    if not video_path:
        return

    try:
        # Extract frames at intervals
        print("Reading video")
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0 and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.release()
        print("Finished reading video")

        # Process faces
        for frame in frames:
            print("Processing frames...")
            encodings = face_recognition.face_encodings(frame)
            for encoding in encodings:
                if not match_and_save(encoding, performance, face_id_counter):
                    save_new_face(encoding, performance, face_id_counter, lock)
    except Exception as e:
        print(f"Error processing video {url}: {e}")
    finally:
        print("Done with another video")
        print("Current files in faces dataset:", os.listdir(DATASET_DIR))
        if os.path.exists(video_path):
            os.remove(video_path)

def match_and_save(encoding, performance, face_id_counter):
    # Check if a face matches any in the dataset
    print("Checking for a match")
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".npy"):
            stored_encoding = np.load(os.path.join(DATASET_DIR, file), allow_pickle=True)
            if face_recognition.compare_faces([stored_encoding], encoding, tolerance=0.6)[0]:
                print("Found a match")
                np.save(os.path.join(DATASET_DIR, file), stored_encoding)
                res[face_id_counter.value, ]
                return True
    print("No match")
    return False

def save_new_face(encoding, performance, face_id_counter, lock):
    # Save a new face encoding to the dataset
    print("Saving new face")
    with lock:
        face_id_counter.value += 1
        face_path = os.path.join(DATASET_DIR, f"face_{face_id_counter.value}.npy")
        print(f"Saving face encoding to: {face_path}")
        np.save(face_path, encoding)
        res[face_id_counter.value, 0] = performance
        print(f"New face saved as face_{face_id_counter.value}")

def performance_average():
    print("Moving on to performance averaging...")
    FACE_AVG = []
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".npy"):
            face = np.load(os.path.join(DATASET_DIR, file), allow_pickle=True)
            print("Face file:", file)
            avg = np.mean(face[128:])
            FACE_AVG.append({'Name': file, 'Avg_performance': avg})
            facelist = face.tolist()
            facelist.append(avg)
            face = np.array(facelist)
            np.save(os.path.join(DATASET_DIR, file), face)
            print("Performance:", avg)
    avg_file = os.path.join(DATASET_DIR, f"Performance.npy")
    np.save(avg_file, np.asarray(FACE_AVG))
    print("Done saving the performance file")

def main(video_urls, performancevalues):
    manager = Manager()
    face_id_counter = manager.Value('i', 0)
    lock = manager.Lock()
    with Pool(cpu_count()) as pool:
        print("Starting Starmapping")
        print("Current working directory:", os.getcwd())
        pool.starmap(process_video, [(url, performance, face_id_counter, lock) for url, performance in zip(video_urls, performancevalues)])
        print("All faces identified and registered")
        performance_average()

if __name__ == "__main__":
    ds = pd.read_csv(r'C:\Users\Sharmeen\Desktop\FuelGrowth_Assignment\Assignment_Data.csv')
    video_urls = ds['Video URL'].tolist() 
    performancevalues = ds['Performance'].tolist()
    main(video_urls, performancevalues)