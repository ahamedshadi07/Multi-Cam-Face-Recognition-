# generate_embeddings.py

import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras_facenet import FaceNet
from numpy.linalg import norm
from sklearn.decomposition import PCA

# Enable GPU acceleration
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# Load FaceNet model
embedder = FaceNet()

dataset_path = "f2images"
if not os.path.exists(dataset_path):
    print("Dataset folder NOT FOUND!")
    exit()

face_names = []
face_embeddings = []

def normalize_embedding(embedding):
    return embedding / (norm(embedding) + 1e-10)

# Process images
for filename in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f" Skipping {filename} (Invalid image)")
        continue

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = embedder.extract(rgb_img, threshold=0.90)

    if len(faces) == 0:
        print(f" No face detected in {filename}")
        continue

    for face in faces:
        embedding = normalize_embedding(face["embedding"])
        name = os.path.splitext(filename)[0]
        face_names.append(name)
        face_embeddings.append(embedding)
        print(f" Processed: {filename}")

face_embeddings = np.array(face_embeddings)

if face_embeddings.shape[0] == 0:
    print(" No embeddings generated. Check your images.")
    exit()

# PCA reduction
pca_components = min(128, face_embeddings.shape[0])
pca = PCA(n_components=pca_components)
face_embeddings = pca.fit_transform(face_embeddings)

# Save PCA and embeddings
with open("pca_model.pkl", "wb") as f:
    pickle.dump(pca, f)

with open("stored_embeddings.pkl", "wb") as f:
    pickle.dump({"names": face_names, "embeddings": face_embeddings}, f)

print("✅ Embeddings saved successfully!")
