import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2 as cv
import numpy as np

model_path = r'C:\\Users\\alexo\Documents\\projects\\pose_estimation\\pose_landmarker_lite.task'
BaseOptions = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=BaseOptions,running_mode=vision.RunningMode.VIDEO, output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image




if __name__ == '__main__':
    video = cv.VideoCapture(0)

    frame_timestamp_ms = video.get(cv.CAP_PROP_POS_MSEC)
    calc_timestamps = [0.0]
    while(True):
        ret, frame = video.read()
        cv.imshow('frame',frame)
        np_frame = np.stack(frame, axis = 0)
        frame_timestamp_ms = video.get(cv.CAP_PROP_POS_MSEC)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_frame)
        result = detector.detect_for_video(mp_image, int(frame_timestamp_ms))
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
        cv.imshow('frame 2',cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv.destroyAllWindows