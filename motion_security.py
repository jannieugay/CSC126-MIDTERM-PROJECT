"""
Motion-Triggered Security Camera
CSC 126 - Computer Vision Midterm Project

Detects movement from a webcam or a recorded video and raises an intruder alert.

OpenCV topics used (from the laboratory list):
  a. Image Operations ............ resize, grayscale, Gaussian blur, drawing text/boxes
  b. Image Segmentation .......... MOG2 background subtraction, thresholding, morphology
  e. Motion and Object Tracking .. contour extraction, bounding boxes, centroid trail
  c. Haar Cascade Classifier ..... (optional, --faces) face detection inside moving regions

Controls (while the window is open):
  q  - quit
  s  - save a manual screenshot
  space - pause / resume
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PROCESS_WIDTH = 480        # frames are resized to this width for speed
WARMUP_FRAMES = 30         # frames used to let the background model learn
TRAIL_LENGTH = 40          # how many past centroids to draw as a motion trail

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


# ----------------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Motion-triggered security camera using OpenCV."
    )
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0) or path to a video file. Default: 0 (webcam)."
    )
    parser.add_argument(
        "--min-area", type=int, default=1200,
        help="Smallest contour area (pixels) counted as motion. Default: 1200."
    )
    parser.add_argument(
        "--cooldown", type=float, default=3.0,
        help="Seconds to wait between saved snapshots. Default: 3."
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Folder for snapshots, event log and video. Default: output."
    )
    parser.add_argument(
        "--faces", action="store_true",
        help="Also run Haar cascade face detection inside moving regions."
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save the annotated result as output/annotated.mp4."
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Run without opening a window (useful on servers)."
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Input / output helpers
# ----------------------------------------------------------------------------
def open_source(source):
    """Open a webcam (if source is a number) or a video file."""
    if source.isdigit():
        capture = cv2.VideoCapture(int(source))
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"[ERROR] Could not open source: {source}")
        sys.exit(1)
    return capture


def prepare_output_folders(output_dir):
    """Create the output folders and the CSV log (with header) if needed."""
    snapshot_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "events.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "frame_number", "motion_regions", "largest_area", "snapshot"]
            )
    return snapshot_dir, log_path


def log_event(log_path, frame_number, regions, snapshot_name):
    """Append one motion event to the CSV log."""
    largest = max(area for _, area in regions)
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            frame_number,
            len(regions),
            int(largest),
            snapshot_name,
        ])


def save_snapshot(frame, snapshot_dir, prefix="motion"):
    """Save a timestamped JPEG and return its file name."""
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(snapshot_dir, name), frame)
    return name


# ----------------------------------------------------------------------------
# Image operations (topic a)
# ----------------------------------------------------------------------------
def resize_frame(frame, width=PROCESS_WIDTH):
    """Resize a frame to a fixed width, keeping the aspect ratio."""
    height = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, height))


def preprocess(frame):
    """Convert to grayscale and blur to reduce noise before subtraction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


# ----------------------------------------------------------------------------
# Segmentation (topic b)
# ----------------------------------------------------------------------------
def create_background_subtractor():
    """MOG2 learns the static background and flags anything that changes."""
    return cv2.createBackgroundSubtractorMOG2(
        history=300, varThreshold=40, detectShadows=True
    )


def get_motion_mask(subtractor, gray):
    """Return a clean binary mask where white pixels are moving."""
    fg_mask = subtractor.apply(gray)
    # MOG2 marks shadows as gray (127); keep only strong foreground (255).
    _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    # Remove small specks, then fill holes in the moving object.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask


# ----------------------------------------------------------------------------
# Motion / object tracking (topic e)
# ----------------------------------------------------------------------------
def find_motion_regions(mask, min_area):
    """Find contours in the mask and keep the ones large enough to be real motion.

    Returns a list of ((x, y, w, h), area).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            regions.append((cv2.boundingRect(contour), area))
    return regions


def update_trail(trail, regions):
    """Store the centroid of the largest moving region to draw a motion trail."""
    if regions:
        (x, y, w, h), _ = max(regions, key=lambda r: r[1])
        trail.append((x + w // 2, y + h // 2))
    else:
        trail.clear()


# ----------------------------------------------------------------------------
# Haar cascade (topic c, optional)
# ----------------------------------------------------------------------------
def load_face_cascade():
    path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        print("[WARNING] Could not load Haar cascade; face detection disabled.")
        return None
    return cascade


def detect_faces(cascade, gray, regions):
    """Run the face detector only inside moving regions. Returns face boxes."""
    faces = []
    for (x, y, w, h), _ in regions:
        roi = gray[y:y + h, x:x + w]
        if roi.shape[0] < 40 or roi.shape[1] < 40:
            continue
        found = cascade.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (fx, fy, fw, fh) in found:
            faces.append((x + fx, y + fy, fw, fh))
    return faces


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------
def draw_results(frame, regions, faces, trail, alarm_on, frame_number, fps):
    """Draw boxes, labels, trail and the alert banner on a copy of the frame."""
    output = frame.copy()

    for (x, y, w, h), area in regions:
        cv2.rectangle(output, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(output, f"Motion {int(area)}", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)

    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), YELLOW, 2)
        cv2.putText(output, "Face", (x, max(y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)

    # Motion trail (line through recent centroids)
    for i in range(1, len(trail)):
        cv2.line(output, trail[i - 1], trail[i], (255, 128, 0), 2)

    # Alert banner
    if alarm_on:
        cv2.rectangle(output, (0, 0), (output.shape[1], 36), RED, -1)
        cv2.putText(output, "INTRUDER DETECTED", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    else:
        cv2.putText(output, "Status: Clear", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

    # Timestamp + info at the bottom
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(output, f"{stamp}  |  Frame {frame_number}  |  {fps:.1f} FPS",
                (10, output.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return output


def build_display(annotated, mask):
    """Place the annotated frame and the motion mask side by side."""
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(mask_bgr, "Foreground Mask", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return np.hstack((annotated, mask_bgr))


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    capture = open_source(args.source)
    snapshot_dir, log_path = prepare_output_folders(args.output_dir)

    subtractor = create_background_subtractor()
    face_cascade = load_face_cascade() if args.faces else None

    trail = deque(maxlen=TRAIL_LENGTH)
    writer = None
    frame_number = 0
    last_alert_time = 0.0
    alarm_until = 0.0          # keep the red banner on for a short time
    paused = False
    prev_time = time.time()
    fps = 0.0

    print("Running... press 'q' to quit, 's' for screenshot, SPACE to pause.")

    while True:
        if not paused:
            ok, raw_frame = capture.read()
            if not ok:
                print("End of video or camera error. Stopping.")
                break
            frame_number += 1

            frame = resize_frame(raw_frame)
            gray = preprocess(frame)
            mask = get_motion_mask(subtractor, gray)

            # Ignore the first frames while the background model is learning
            regions = []
            if frame_number > WARMUP_FRAMES:
                regions = find_motion_regions(mask, args.min_area)
            update_trail(trail, regions)

            faces = []
            if face_cascade is not None and regions:
                faces = detect_faces(face_cascade, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), regions)

            # FPS (smoothed)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            # Alert logic: snapshot + log, limited by the cooldown
            if regions:
                alarm_until = now + 1.0
                if now - last_alert_time >= args.cooldown:
                    last_alert_time = now
                    annotated_for_save = draw_results(
                        frame, regions, faces, trail, True, frame_number, fps)
                    name = save_snapshot(annotated_for_save, snapshot_dir)
                    log_event(log_path, frame_number, regions, name)
                    print(f"[ALERT] Motion at frame {frame_number} -> {name}")

            annotated = draw_results(
                frame, regions, faces, trail, now < alarm_until, frame_number, fps)
            display = build_display(annotated, mask)

            if args.save_video:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        os.path.join(args.output_dir, "annotated.mp4"), fourcc, 20.0, (w, h))
                writer.write(annotated)

        if not args.no_display:
            cv2.imshow("Motion Security Camera  (annotated | mask)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                name = save_snapshot(annotated, snapshot_dir, prefix="manual")
                print(f"[SAVED] {name}")
            elif key == ord(" "):
                paused = not paused

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Results saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
