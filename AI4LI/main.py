import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import joblib



base_options = python.BaseOptions(model_asset_path='./hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

scaler = joblib.load('./scaler.joblib')  # Load the scaler
svm = joblib.load('./svm_model.joblib')  # Load the SVM model

classes = {'A':0,'B':1,'C':2, 'D':3,'E':4,'F':5,'G':6,'H':7,'H_multiple':8,'I':9,'L':10,'empty':11}
classes_from_index = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'H_multiple',9:'I',10:'L',11:'empty'}

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  left_x = [0.0] * 21
  left_y = [0.0] * 21
  left_z = [0.0] * 21
  right_x = [0.0] * 21
  right_y = [0.0] * 21
  right_z = [0.0] * 21
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    # print("Proto:------",hand_landmarks_proto)
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape

    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    z_coordinates = [landmark.z for landmark in hand_landmarks]
    
    if handedness[0].category_name == 'Left':
        left_x = x_coordinates
        left_y = y_coordinates
        left_z = z_coordinates
    elif handedness[0].category_name == 'Right':
        right_x = x_coordinates
        right_y = y_coordinates
        right_z = z_coordinates
      
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

  
    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  x_cords = left_x + right_x
  y_cords = left_y + right_y
  z_cords = left_z + right_z
  return annotated_image, x_cords, y_cords, z_cords

def detect_and_draw(img):
   
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    detection_result = detector.detect(image)

    annotated_image, x, y ,z = draw_landmarks_on_image(image.numpy_view(), detection_result)

    if np.sum(x) != 0:
        
        feats = [x + y + z]
        feats = scaler.transform(feats)  # Scale the features using the same scaler
        predict = svm.predict(feats)

        annotated_image= cv2.putText(annotated_image, f"{classes_from_index[predict[0]]}",
                    (150, 150), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    return annotated_image

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    # Write the frame to the output file
    out.write(frame)
  
    # Display the captured frame

    annotated_frame = detect_and_draw(frame)
    cv2.imshow('Camera', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
