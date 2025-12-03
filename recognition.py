import cv2
import numpy as np
import pickle
import threading
import tensorflow as tf
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datetime import datetime
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import time

# Load FaceNet embedder
embedder = FaceNet()

# Load stored embeddings and PCA
with open("stored_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

known_names = data["names"]
known_embeddings = list(data["embeddings"])
unregistered_embeddings = []
registered_names_counter = len(set(known_names))

tracker_dict = {}
last_logged_times = {}
attendance_session = set()

camera_locations = {
    0: "Entrance",
    1: "Classroom"
}


# Normalize embedding
def normalize_embedding(embedding):
    norm_val = np.linalg.norm(embedding)
    return embedding / (norm_val + 1e-10)


# Log every face detected and processed
def log_detection(face_id, identity, confidence, frame):
    print(f"[Detection Log] Face ID: {face_id}, Identity: {identity}, Confidence: {confidence:.2f}")
    cv2.putText(frame, f"Detected: {identity} ({confidence:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Visual feedback for unknown faces
    if identity == "Unknown":
        cv2.putText(frame, "Unknown Face", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    else:
        cv2.putText(frame, f"Known Face: {identity}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



# Recognize face with dynamic threshold
def recognize_face(embedding, frame):
    try:
        embedding = normalize_embedding(embedding)
        embedding_pca = pca.transform([embedding])
        similarities = cosine_similarity(embedding_pca, known_embeddings)[0]
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]

        mean_sim = np.mean(similarities)
        dynamic_threshold = max(0.6, min(0.85, mean_sim - 0.05))  # Adaptive threshold

        if confidence > dynamic_threshold:
            identity = known_names[best_match_index]
            log_detection(best_match_index, identity, confidence, frame)
            return identity, confidence
        else:
            identity = "Unknown"
            log_detection(best_match_index, identity, confidence, frame)
            if len(unregistered_embeddings) < 100:
                unregistered_embeddings.append(embedding)
            return identity, confidence
    except Exception as e:
        print(f"[Recognition Error]: {e}")
        return "Unknown", None



# Try to register new faces with DBSCAN
def try_register_new_faces():
    global registered_names_counter, known_names, known_embeddings

    if len(unregistered_embeddings) < 15:
        return

    print("[DBSCAN] Trying to register new faces...")

    # Check for new faces using DBSCAN (clustering of unregistered embeddings)
    X = pca.transform(unregistered_embeddings)
    clustering = DBSCAN(eps=0.6, min_samples=5).fit(X)
    labels = clustering.labels_
    print("[DBSCAN] Labels:", labels)

    for label in set(labels):
        if label == -1:
            continue  # Skip noise (unclustered faces)

        indices = [i for i, lbl in enumerate(labels) if lbl == label]
        cluster_embeddings = [X[i] for i in indices]
        avg_embedding = np.mean(cluster_embeddings, axis=0)

        # Check similarity of the clustered face to existing known faces
        max_similarity = cosine_similarity([avg_embedding], known_embeddings)[0]
        best_match_index = np.argmax(max_similarity)
        confidence = max_similarity[best_match_index]

        if confidence < 0.75:  # If not similar enough to existing faces, add as new
            new_name = f"Person_{registered_names_counter + 1}"
            known_names.append(new_name)
            known_embeddings.append(avg_embedding)
            registered_names_counter += 1
            print(f"[DBSCAN] Registered new face as {new_name}")
        else:
            print(f"[DBSCAN] Clustering face is too similar to {known_names[best_match_index]}. Skipping.")

    unregistered_embeddings.clear()

    # Save updated embeddings
    with open("stored_embeddings.pkl", "wb") as f:
        pickle.dump({"names": known_names, "embeddings": known_embeddings}, f)


# Log attendance
def log_attendance(name, camera_id=0, cooldown_minutes=5):
    if name == "Unknown" or name in attendance_session:
        return

    now = datetime.now()
    current_time = time.time()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    key = f"{name}_cam{camera_id}"
    if key in last_logged_times:
        elapsed = current_time - last_logged_times[key]
        if elapsed < cooldown_minutes * 60:
            return

    last_logged_times[key] = current_time
    attendance_session.add(name)

    cam_location = camera_locations.get(camera_id, f"Camera_{camera_id}")
    filename = r"D:\\Documents\\Final year project\\Master_Attendance.csv"

    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Name,Date,Time,Camera Location\n")

    with open(filename, "a") as f:
        f.write(f"{name},{date_str},{time_str},{cam_location}\n")


# Process each camera
def process_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    tracker = DeepSort(max_age=5)
    tracker_dict[cam_index] = tracker

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = embedder.extract(rgb_frame, threshold=0.9)

        bboxes = []
        embeddings = []

        for face in detections:
            box = face["box"]
            emb = face["embedding"]
            x, y, w, h = box
            bbox = [x, y, x + w, y + h]
            bboxes.append((bbox, 1.0, 'face'))
            embeddings.append(emb)

        tracks = tracker.update_tracks(bboxes, frame=frame)

        for i, track in enumerate(tracks):
            if not track.is_confirmed() or track.track_id is None:
                continue

            track_id = track.track_id
            embedding = embeddings[i] if i < len(embeddings) else None
            if embedding is None:
                continue

            identity, conf = recognize_face(embedding, frame)
            log_attendance(identity, camera_id=cam_index)

            l, t, r, b = track.to_ltrb()
            label = f"Cam {cam_index} | Track ID {track_id}: {identity} ({conf:.2f})" if conf else identity
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        try_register_new_faces()

        cv2.imshow(f"Camera {cam_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Start threads for cameras
camera_threads = []
for i in range(2):  # Update 2 to number of cameras
    t = threading.Thread(target=process_camera, args=(i,))
    t.start()
    camera_threads.append(t)

for t in camera_threads:
    t.join()

print(" Final Registered Names:", known_names)
print(" Total Embeddings Saved:", len(known_embeddings))
