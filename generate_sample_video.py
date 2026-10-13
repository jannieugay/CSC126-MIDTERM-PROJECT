"""
Generates a short synthetic test video (samples/sample_motion.mp4).

The scene is a static background. After about 2 seconds a "person" (a moving
rectangle with a head) walks across the frame, so you can test the motion
detector without a webcam. Replace it with your own recorded video if you like.
"""

import os

import cv2
import numpy as np

WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 20, 12
OUTPUT = os.path.join("samples", "sample_motion.mp4")


def make_background():
    """A simple fixed 'room': gray wall, darker floor, and a door."""
    bg = np.full((HEIGHT, WIDTH, 3), 170, dtype=np.uint8)
    cv2.rectangle(bg, (0, 340), (WIDTH, HEIGHT), (90, 90, 90), -1)      # floor
    cv2.rectangle(bg, (480, 120), (590, 340), (60, 80, 120), -1)         # door
    cv2.rectangle(bg, (60, 100), (200, 200), (200, 220, 240), -1)        # window
    return bg


def main():
    os.makedirs("samples", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT, fourcc, FPS, (WIDTH, HEIGHT))
    background = make_background()
    rng = np.random.default_rng(0)

    for i in range(FPS * SECONDS):
        frame = background.copy()
        t = i / FPS

        # Walker enters at 2s, crosses left->right by 7s, then leaves
        if 2.0 <= t <= 8.0:
            x = int(-60 + (t - 2.0) / 6.0 * (WIDTH + 120))
            cv2.rectangle(frame, (x, 230), (x + 60, 360), (30, 30, 160), -1)  # body
            cv2.circle(frame, (x + 30, 205), 22, (120, 170, 220), -1)         # head

        # Light camera noise so the background is not perfectly static
        noise = rng.integers(-3, 4, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
