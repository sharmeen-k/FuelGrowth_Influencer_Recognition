# FuelGrowth: Influencer Recognition and Ranking

This project processes a dataset of videos to identify influencers by detecting unique faces and analyzing their engagement metrics. It includes tools for video processing, face recognition, and ranking based on performance metrics.

## Features
**Face Detection and Encoding:** Detects unique faces from video frames and encodes them for further analysis.  
**Influencer Performance Analysis:** Analyzes the raw performance data for every encoding.  
**Ranking System:** Creates a ranked list of influencers based on engagement data.

## Files and Structure
**Assignment_Data.csv:** Dataset used. Contains video URLs and performance metrics for analysis.  
**main.py:** Main script that processes videos, detects faces, and manages performance records in face_dataset.  
**face_dataset:** Directory for storing encodings of all the faces identified during the operation of main.py. Also stores the performance.csv file containing performance data and the URL for each unique face in every video.  
**face_dataset_processing.ipynb:** Jupyter Notebook for final face dataset processing and output.  
**top_influencer.jpg:** Image of the highest performing influencer.  
**Influencer_Ranking.csv:** Outputs the ranked list of influencers based on engagement performance.  

## Workflow
**1. Dataset Initialization:**  
Initializes performance.csv to store performance data for encodings. Assignment_Data.csv provides video links and their associated performance data.  

**2. Video Processing:**
Extracts frames from videos.
Detects and encodes faces using face_recognition.
Matches or saves new faces in the face_dataset directory.

**3. Performance Recording:**
Saves engagement metrics for identified faces in performance.csv.

**4. Ranking:**
Generates Influencer_Ranking.csv to summarize influencer performance.

## Setup and Execution
**Prerequisites**
Python 3.7 or higher  
numpy, pandas, opencv-python, face_recognition

**Install dependencies using pip:**

```bash
pip install -r requirements.txt
```

**Execution**
Run the main script to process the dataset:

```bash
python main.py
```

Ensure Assignment_Data.csv is in the working directory.

## Outputs  
**top_influencer.jpg:** Image of the best-performing influencer.  
**Influencer_Ranking.csv:** Final ranking of influencers based on their performance. Final processed dataset.  
**face_dataset/performance.csv:** Stores engagement metrics for each identified face with URL. Raw output dataset.  

## Future Enhancements
Optimization for parallel processing on large video datasets.  
Improved ranking models incorporating advanced metrics.

## Acknowledgments
This project uses the following libraries:

OpenCV for video processing.  
face_recognition for facial detection and encoding.  
Pandas for data manipulation.
