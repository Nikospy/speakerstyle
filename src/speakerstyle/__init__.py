"""speakerstyle: one halftone cut-out house style for portraits and objects."""
import os

# OpenCV prints a harmless backend notice on every face detection; keep the terminal readable.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
