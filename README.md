Multi-Camera Face Recognition System

A real-time face recognition and attendance system using FaceNet, DeepSORT, PCA, and DBSCAN, capable of detecting, tracking, recognizing, and auto-registering faces across multiple camera feeds.
Features

    • Real-time face detection and recognition using FaceNet
    • PCA-reduced embeddings for faster similarity matching
    • Cosine similarity with adaptive threshold for accurate recognition
    • Supports multiple camera streams using threading
    • DeepSORT-based object tracking for stable Track IDs
    • Automatic clustering of unknown faces using DBSCAN
    • Auto-registration of new persons based on unknown clusters
    • Attendance logging with date, time, and camera location
    
Project Structure

    Multi-Camera-Face-Recognition/
    • generate_embeddings.py
    • recognition.py
    • pca_model.pkl
    • stored_embeddings.pkl
    • README.md
    
How the System Works
    1. Embedding Generation
    • Face images are processed to extract FaceNet embeddings.
    • Embeddings are normalized and reduced using PCA.
    • All embeddings and corresponding names are saved in .pkl files.
    2. Real-Time Multi-Camera Recognition
    • Each camera is started in a separate thread.
    • Faces are detected and embeddings are generated in real time.
    • DeepSORT tracks each face and assigns consistent Track IDs.
    • Faces are matched using cosine similarity.
    • Unknown faces are stored and grouped using DBSCAN.
    • Each new unknown cluster is automatically assigned a name such as Person_1, Person_2, etc.
    • Attendance is recorded in Master_Attendance.csv automatically.
    
Installation Steps
1. Clone the repository:
git clone https://github.com/ahamedshadi07/Multi-Camera-Face-Recognition-.git

3. Open the project folder:
cd Multi-Camera-Face-Recognition-

5. Install the required packages:
pip install -r requirements.txt
Usage Instructions
To Generate Embeddings:
python generate_embeddings.py
To Start Multi-Camera Recognition:
python recognition.py

Additional Notes
    • Compatible with both USB and IP cameras.
    • You can change the number of cameras in the script:
    for i in range(2):
    • Attendance CSV is created automatically if it does not exist.
    License
MIT License
Author
Ahamed Shadi
