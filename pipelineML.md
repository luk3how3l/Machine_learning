# **Architecture Document: In-Memory Multi-Image Preprocessing Pipeline**

### **1\. Project Overview**

The goal is to process a large volume of images distributed across 20+ ZIP archives (e.g., images(1).zip, images(2).zip) on a single machine. To maximize speed, this pipeline bypasses traditional disk extraction and reads images directly from the archives into memory for person detection and cropping using **YOLOv8s**.

### **2\. System Architecture**

This pipeline follows a **Data-Parallel Multiprocessing** model. Instead of a linear "extract then process" sequence, the system treats each ZIP file as a concurrent task.

* **Orchestrator (Main Process):** Initializes the environment, creates output directories, and manages a pool of worker processes.  
* **Worker Processes (Sub-processes):** Each worker is assigned one or more ZIP files.  
  * **Internal Loop:** Since each ZIP file (e.g., images(1).zip) contains multiple images, the worker opens the archive once and iterates through every image file inside it.  
  * **In-Memory Stream:** Each image is read as a byte stream directly from the ZIP into a RAM buffer.  
  * **GPU Inference:** Each worker maintains its own instance of yolov8s.pt to perform local inference without resource contention.  
* **Storage (Disk I/O):** Only the final "vetted" crops or filtered images are written to the hard drive, reducing disk wear and I/O latency.

---

### **3\. Step-by-Step Implementation Instructions**

#### **Step 1: Environment Preparation**

Install the high-performance computer vision stack.

Bash  
pip install ultralytics opencv-python-headless numpy

#### **Step 2: Configure System Initialization**

When working with GPUs and multiprocessing, you must use the spawn method to ensure the CUDA context is properly initialized in each child process.

* **Action:** Add multiprocessing.set\_start\_method('spawn', force=True) inside your if \_\_name\_\_ \== "\_\_main\_\_": block.

#### **Step 3: Define the Worker Function (process\_zip\_archive)**

This function handles the logic for a single ZIP file containing multiple images.

1. **Initialize Model:** Load YOLO("yolov8s.pt") inside this function so each worker has its own copy.  
2. **Open Archive:** Open the ZIP file using zipfile.ZipFile(zip\_path, 'r').  
3. **Iterate Internal Files:** Loop through zip\_ref.namelist().  
   * **Filter:** Check if the file extension is an image (e.g., .jpg, .png).  
4. **In-Memory Decode:**  
   * Read image bytes: img\_bytes \= zip\_ref.read(image\_name).  
   * Convert to NumPy: Use np.frombuffer(img\_bytes, np.uint8) followed by cv2.imdecode() to get a usable image array without saving a temporary file to disk.  
5. **Run YOLOv8s Inference:**  
   * Run model.predict(image, classes=). Class ID 0 is specifically for "person" in the COCO dataset used by YOLOv8.  
6. **Filter and Crop Logic:**  
   * **Person Detected:** If len(results.boxes) \> 0, extract the bounding box coordinates $(x1, y1, x2, y2)$. Use NumPy slicing to crop: cropped \= image\[int(y1):int(y2), int(x1):int(x2)\]. Save to output/Person/.  
   * **No Person:** Save the original image to output/No\_Person/.

#### **Step 4: Setup the Main Orchestrator**

1. **Directory Setup:** Use os.makedirs to create your destination folders.  
2. **Collect Inputs:** Create a list of all your ZIP file paths.  
3. **Launch Pool:** Use concurrent.futures.ProcessPoolExecutor to map your worker function across the list of ZIP files.

---

### **4\. Performance & Hardware Optimization**

| Optimization | Benefit |
| :---- | :---- |
| **Direct RAM Streaming** | Saves up to 50% of total time by skipping "Extraction to Disk" cycles. |
| **Class Filtering** | Using classes= in the predict call forces the model to ignore 79 other object types, reducing post-processing overhead. |
| **Worker Model Loading** | Prevents "Pickling Errors" and "Global Interpreter Lock (GIL)" bottlenecks by isolating the model in each process memory. |
| **Batching** | If a single ZIP has thousands of images, have the worker group 16 images into a list before calling model.predict(). This keeps the GPU cores fully saturated. |

### **5\. Final Folder Structure**

Your script will transform your raw data into this ready-to-train format:

/Project\_Root

├── raw\_zips/

│ ├── images(1).zip (contains img1, img2, img3...)

│ └── images(2).zip

└── output/

├── Person/ (Cropped images of detected people)

└── No\_Person/ (Images where no one was found)

# **Architecture Document: In-Memory Multi-Image Preprocessing Pipeline**

### **1\. Project Overview**

The goal is to process a large volume of images distributed across 20+ ZIP archives (e.g., images(1).zip, images(2).zip) on a single machine. To maximize speed, this pipeline bypasses traditional disk extraction and reads images directly from the archives into memory for person detection and cropping using **YOLOv8s**.

### **2\. System Architecture**

This pipeline follows a **Data-Parallel Multiprocessing** model. Instead of a linear "extract then process" sequence, the system treats each ZIP file as a concurrent task.

* **Orchestrator (Main Process):** Initializes the environment, creates output directories, and manages a pool of worker processes.  
* **Worker Processes (Sub-processes):** Each worker is assigned one or more ZIP files.  
  * **Internal Loop:** Since each ZIP file (e.g., images(1).zip) contains multiple images, the worker opens the archive once and iterates through every image file inside it.  
  * **In-Memory Stream:** Each image is read as a byte stream directly from the ZIP into a RAM buffer.  
  * **GPU Inference:** Each worker maintains its own instance of yolov8s.pt to perform local inference without resource contention.  
* **Storage (Disk I/O):** Only the final "vetted" crops or filtered images are written to the hard drive, reducing disk wear and I/O latency.

---

### **3\. Step-by-Step Implementation Instructions**

#### **Step 1: Environment Preparation**

Install the high-performance computer vision stack.

Bash  
pip install ultralytics opencv-python-headless numpy

#### **Step 2: Configure System Initialization**

When working with GPUs and multiprocessing, you must use the spawn method to ensure the CUDA context is properly initialized in each child process.

* **Action:** Add multiprocessing.set\_start\_method('spawn', force=True) inside your if \_\_name\_\_ \== "\_\_main\_\_": block.

#### **Step 3: Define the Worker Function (process\_zip\_archive)**

This function handles the logic for a single ZIP file containing multiple images.

1. **Initialize Model:** Load YOLO("yolov8s.pt") inside this function so each worker has its own copy.  
2. **Open Archive:** Open the ZIP file using zipfile.ZipFile(zip\_path, 'r').  
3. **Iterate Internal Files:** Loop through zip\_ref.namelist().  
   * **Filter:** Check if the file extension is an image (e.g., .jpg, .png).  
4. **In-Memory Decode:**  
   * Read image bytes: img\_bytes \= zip\_ref.read(image\_name).  
   * Convert to NumPy: Use np.frombuffer(img\_bytes, np.uint8) followed by cv2.imdecode() to get a usable image array without saving a temporary file to disk.  
5. **Run YOLOv8s Inference:**  
   * Run model.predict(image, classes=). Class ID 0 is specifically for "person" in the COCO dataset used by YOLOv8.  
6. **Filter and Crop Logic:**  
   * **Person Detected:** If len(results.boxes) \> 0, extract the bounding box coordinates $(x1, y1, x2, y2)$. Use NumPy slicing to crop: cropped \= image\[int(y1):int(y2), int(x1):int(x2)\]. Save to output/Person/.  
   * **No Person:** Save the original image to output/No\_Person/.

#### **Step 4: Setup the Main Orchestrator**

1. **Directory Setup:** Use os.makedirs to create your destination folders.  
2. **Collect Inputs:** Create a list of all your ZIP file paths.  
3. **Launch Pool:** Use concurrent.futures.ProcessPoolExecutor to map your worker function across the list of ZIP files.

---

### **4\. Performance & Hardware Optimization**

| Optimization | Benefit |
| :---- | :---- |
| **Direct RAM Streaming** | Saves up to 50% of total time by skipping "Extraction to Disk" cycles. |
| **Class Filtering** | Using classes= in the predict call forces the model to ignore 79 other object types, reducing post-processing overhead. |
| **Worker Model Loading** | Prevents "Pickling Errors" and "Global Interpreter Lock (GIL)" bottlenecks by isolating the model in each process memory. |
| **Batching** | If a single ZIP has thousands of images, have the worker group 16 images into a list before calling model.predict(). This keeps the GPU cores fully saturated. |

### **5\. Final Folder Structure**

Your script will transform your raw data into this ready-to-train format:

/Project\_Root

├── raw\_zips/

│ ├── images(1).zip (contains img1, img2, img3...)

│ └── images(2).zip

└── output/

├── Person/ (Cropped images of detected people)

└── No\_Person/ (Images where no one was found)
