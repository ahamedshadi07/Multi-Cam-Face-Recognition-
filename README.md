# Multi-Cam-Face-Recognition-
Real-time multi-camera face recognition system with detection, embeddings, and attendance generation.

Overview

A real-time face recognition system that works across multiple camera feeds.

Uses FaceNet embeddings, PCA, DeepSORT tracking, and DBSCAN clustering.

Automatically detects, tracks, recognizes, and registers new faces.

Attendance is logged with camera location and timestamps.

Features

Face detection and recognition using FaceNet

PCA for fast and efficient embedding comparison

Similarity check using cosine similarity

Multi-camera real-time processing using threading

DeepSORT tracking for stable and consistent Track IDs

Automatic unknown-face grouping using DBSCAN clustering

Auto-registration of new persons when unknown clusters repeat

Attendance logging with date, time, and camera source
