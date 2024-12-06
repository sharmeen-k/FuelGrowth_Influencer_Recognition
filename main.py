import os
import cv2
import face_recognition
import numpy as np
import pandas as pd
import csv
from urllib.request import urlopen
from multiprocessing import Pool, Manager, cpu_count

# Directory to store face dataset
DATASET_DIR = "face_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
# CSV file to store performance
PERFORMANCE_CSV = os.path.join(DATASET_DIR, "performance.csv")

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
                    save_new_face(encoding, performance, url, face_id_counter, lock)
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
                save_performance(face_id_counter.value, performance)
                return True
    print("No match")
    return False

def save_new_face(encoding, performance, url, face_id_counter, lock):
    # Save a new face encoding to the dataset
    print("Saving new face")
    with lock:
        face_id_counter.value += 1
        face_path = os.path.join(DATASET_DIR, f"face_{face_id_counter.value}.npy")
        print(f"Saving face encoding to: {face_path}")
        np.save(face_path, encoding)
        save_performance(face_id_counter.value, performance, url=url)
        print(f"New face saved as face_{face_id_counter.value}")

def initialize_performance_csv():
    # Creates a csv file to record performance for detected faces
    if not os.path.exists(PERFORMANCE_CSV):
        with open(PERFORMANCE_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Face_ID", "Performance", "Video URL"])
            print("New performance csv created")

def save_performance(face_id, performance, url=None):
    # Saves data in the csv file
    print("Saving performance record for", face_id)
    with open(PERFORMANCE_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([face_id, performance, url])

def main(video_urls, performancevalues):
    manager = Manager()
    face_id_counter = manager.Value('i', 0)
    lock = manager.Lock()
    with Pool(cpu_count()) as pool:
        print("Starting Starmapping")
        print("Current working directory:", os.getcwd())
        pool.starmap(process_video, [(url, performance, face_id_counter, lock) for url, performance in zip(video_urls, performancevalues)])
        print("All faces identified and registered")

if __name__ == "__main__":
    ds = pd.read_csv(r'C:\Users\Sharmeen\Desktop\FuelGrowth_Assignment\Assignment_Data.csv')
    video_urls = ds['Video URL'].tolist() 
    performancevalues = ds['Performance'].tolist()
    initialize_performance_csv()
    main(video_urls, performancevalues)