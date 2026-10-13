# Motion-Triggered Security Camera

**CSC 126 – Computer Vision Midterm Project**
Caraga State University, College of Computing and Information Sciences

## Project Description

A Python + OpenCV application that watches a webcam feed (or a recorded video) and
raises an alert when movement is detected. When motion appears, the program draws
bounding boxes around the moving objects, shows an **INTRUDER DETECTED** banner,
saves a timestamped snapshot, and records the event in a CSV log.

**Real-world use case:** a simple, low-cost home or room security monitor that only
records evidence when something actually moves.

## OpenCV Techniques Used

| Topic | What the project does |
|-------|-----------------------|
| **a. Image Operations** | Resize, grayscale conversion, Gaussian blur, drawing boxes/text/timestamps |
| **b. Image Segmentation** | MOG2 background subtraction, thresholding, morphological opening and dilation |
| **e. Motion and Object Tracking** | Contour extraction, area filtering, bounding boxes, centroid motion trail |
| **c. Haar Cascade (optional)** | Face detection inside moving regions using `--faces` |

## Project Structure

```
motion-security-camera/
├── motion_security.py          # main application
├── generate_sample_video.py    # creates samples/sample_motion.mp4
├── requirements.txt
├── README.md
├── samples/                    # sample input video(s)
├── screenshots/                # screenshots of the output
└── output/                     # created on run: snapshots/, events.csv
```

## Setup

1. Install Python 3.8 or newer.
2. (Recommended) create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS / Linux
   ```
3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## How to Run

**Webcam (default):**
```
python motion_security.py
```

**Recorded video:**
```
python generate_sample_video.py          # only once, to create the sample
python motion_security.py --source samples/sample_motion.mp4
```

**With face detection and saving the annotated video:**
```
python motion_security.py --faces --save-video
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--source` | Webcam index or video file path | `0` |
| `--min-area` | Smallest contour (pixels) counted as motion | `1200` |
| `--cooldown` | Seconds between saved snapshots | `3` |
| `--output-dir` | Where snapshots and logs are saved | `output` |
| `--faces` | Enable Haar cascade face detection | off |
| `--save-video` | Save the annotated video | off |
| `--no-display` | Run without a window | off |

### Keyboard Controls

- `q` quit
- `s` save a manual screenshot
- `SPACE` pause / resume

## Output

- **Window:** annotated frame (left) and the foreground mask (right)
- `output/snapshots/` timestamped JPEG for every alert
- `output/events.csv` log with time, frame number, number of regions, and largest area
- `output/annotated.mp4` (only with `--save-video`)

## Limitations

- Sudden lighting changes (lights switched on, clouds) can cause false alarms.
- Very slow movement may be absorbed into the background model.
- A moving camera will trigger detections across the whole frame.
- Haar cascade face detection works best on frontal faces with good lighting.

## Possible Improvements

- Email or SMS notifications when an alert is triggered
- Define a region of interest (ignore areas such as a window or fan)
- Person-vs-object classification with a better detector
- Start/stop video recording around each event

## Members

- (Add your name and your groupmates' names here)
