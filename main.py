import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np

model_path = r'C:\\Users\\alexo\Documents\\projects\\pose_estimation\\pose_landmarker_lite.task'

BaseOption = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(base_options = BaseOption(model_asset_path=model_path), running_mode=VisionRunningMode.VIDEO)

dector = PoseLandmarker.create_from_options(options)

video = cv.VideoCapture(0)

frame_timestamp_ms = video.get(cv.CAP_PROP_POS_MSEC)
calc_timestamps = [0.0]
while(True):
    ret, frame = video.read()
    cv.imshow('frame',frame)
    np_frame = np.stack(frame, axis = 0)
    frame_timestamp_ms = video.get(cv.CAP_PROP_POS_MSEC)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_frame)
    PoseLandmarker.detect_for_video(mp_image, int(frame_timestamp_ms))
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv.destroyAllWindows