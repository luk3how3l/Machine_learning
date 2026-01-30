import os
import zipfile
import cv2
import numpy as np
from ultralytics import YOLO
import concurrent.futures
import multiprocessing
import glob
from tqdm import tqdm
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Assumes the script is run from MLbibs/start01/
INPUT_DIR = "zip"
OUTPUT_DIR = "output"
PERSON_DIR = os.path.join(OUTPUT_DIR, "Person")
NO_PERSON_DIR = os.path.join(OUTPUT_DIR, "No_Person")

def process_zip_archive(zip_path):
    """
    Worker function to process a single zip archive.
    It extracts images in-memory, runs person detection, and saves results.
    """
    model = YOLO("yolov8s.pt")
    person_class_id = 0  # COCO class ID for 'person'

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for image_name in image_files:
                try:
                    # Step 3.4: In-Memory Decode
                    img_bytes = zip_ref.read(image_name)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is None:
                        logging.warning(f"Could not decode image: {image_name} in {zip_path}")
                        continue
                    
                    # Step 3.5: Run YOLOv8s Inference
                    results = model.predict(img, classes=[person_class_id], verbose=False)
                    
                    # Step 3.6: Filter and Crop Logic
                    person_detected = False
                    if results and results[0].boxes:
                        # Process each detected person
                        for i, box in enumerate(results[0].boxes):
                            person_detected = True
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cropped_img = img[y1:y2, x1:x2]
                            
                            base_name = os.path.splitext(os.path.basename(image_name))[0]
                            zip_name = os.path.splitext(os.path.basename(zip_path))[0]
                            output_filename = f"{zip_name}_{base_name}_person_{i}.jpg"
                            cv2.imwrite(os.path.join(PERSON_DIR, output_filename), cropped_img)

                    if not person_detected:
                        base_name = os.path.basename(image_name)
                        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
                        output_filename = f"{zip_name}_{base_name}"
                        cv2.imwrite(os.path.join(NO_PERSON_DIR, output_filename), img)

                except Exception as e:
                    logging.error(f"Error processing image {image_name} in {zip_path}: {e}")
    except zipfile.BadZipFile:
        logging.error(f"Bad zip file: {zip_path}")
    except Exception as e:
        logging.error(f"Unhandled error in worker for {zip_path}: {e}")
        
    return zip_path

def main():
    """
    Main orchestrator to set up directories and manage the processing pool.
    """
    # Step 4.1: Directory Setup
    logging.info("Creating output directories...")
    os.makedirs(PERSON_DIR, exist_ok=True)
    os.makedirs(NO_PERSON_DIR, exist_ok=True)

    # Step 4.2: Collect Inputs
    zip_files = glob.glob(os.path.join(INPUT_DIR, '*.zip'))
    if not zip_files:
        logging.error(f"No ZIP files found in '{INPUT_DIR}' directory.")
        return
    
    logging.info(f"Found {len(zip_files)} ZIP files to process.")

    # Step 4.3: Launch Pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use tqdm to show progress
        list(tqdm(executor.map(process_zip_archive, zip_files), total=len(zip_files), desc="Processing ZIPs"))

    logging.info("Processing complete.")

if __name__ == "__main__":
    # Step 2: Configure System Initialization
    multiprocessing.set_start_method('spawn', force=True)
    main()
