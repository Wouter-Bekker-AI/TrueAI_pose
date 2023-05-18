import cv2
from ultralytics import YOLO
import torch
import math

font = cv2.FONT_HERSHEY_SIMPLEX  # Font for display

# Load YOLOv8 model
model = YOLO('yolov8l-pose.pt')

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)

# Open video capture device
video_path = 0
cap = cv2.VideoCapture(video_path)

# Get video properties
fps, width, height = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Input has W:{width} H:{height} FPS:{fps}')

# Define video writer
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

while cap.isOpened():
    # Read frame from video capture device
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)

        results = model(frame, verbose=False)

        # Visualize the results on the frame if they exist
        if len(results[0].keypoints) > 0:

            # Whether to plot the model frame or not
            annotated_frame = results[0].plot(img=frame)

            # Key the locations of each joint
            keypoints = results[0].keypoints[0].squeeze().tolist()

            nose_x, nose_y = keypoints[0][:2] if keypoints[0][2] > 0.51 else (None, None)
            right_eye_x, right_eye_y = keypoints[1][:2] if keypoints[1][2] > 0.51 else (None, None)
            left_eye_x, left_eye_y = keypoints[2][:2] if keypoints[2][2] > 0.51 else (None, None)
            right_ear_x, right_ear_y = keypoints[3][:2] if keypoints[3][2] > 0.51 else (None, None)
            left_ear_x, left_ear_y = keypoints[4][:2] if keypoints[4][2] > 0.51 else (None, None)
            right_shoulder_x, right_shoulder_y = keypoints[5][:2] if keypoints[5][2] > 0.51 else (None, None)
            left_shoulder_x, left_shoulder_y = keypoints[6][:2] if keypoints[6][2] > 0.51 else (None, None)
            right_elbow_x, right_elbow_y = keypoints[7][:2] if keypoints[7][2] > 0.51 else (None, None)
            left_elbow_x, left_elbow_y = keypoints[8][:2] if keypoints[8][2] > 0.51 else (None, None)
            right_wrist_x, right_wrist_y = keypoints[9][:2] if keypoints[9][2] > 0.51 else (None, None)
            left_wrist_x, left_wrist_y = keypoints[10][:2] if keypoints[10][2] > 0.51 else (None, None)
            right_hip_x, right_hip_y = keypoints[11][:2] if keypoints[11][2] > 0.51 else (None, None)
            left_hip_x, left_hip_y = keypoints[12][:2] if keypoints[12][2] > 0.51 else (None, None)
            right_knee_x, right_knee_y = keypoints[13][:2] if keypoints[13][2] > 0.51 else (None, None)
            left_knee_x, left_knee_y = keypoints[14][:2] if keypoints[14][2] > 0.51 else (None, None)
            right_ankle_x, right_ankle_y = keypoints[15][:2] if keypoints[15][2] > 0.51 else (None, None)
            left_ankle_x, left_ankle_y = keypoints[16][:2] if keypoints[16][2] > 0.51 else (None, None)

            if right_shoulder_y and left_shoulder_y and right_hip_y and left_hip_y:
                shoulder_distance = math.dist((right_shoulder_x, right_shoulder_y), (left_shoulder_x, left_shoulder_y))
                hip_distance = math.dist((right_hip_x, right_hip_y), (left_hip_x, left_hip_y))
                ratio = shoulder_distance/hip_distance
                if ratio > 1.25:
                    cv2.putText(annotated_frame, f'Man', (0, 30), font, 1, (0, 0, 0), 2)
                else:
                    cv2.putText(annotated_frame, f'Woman', (0, 30), font, 1, (0, 0, 0), 2)
                #print(f'{ratio}')

            # Pose profiles
            if right_wrist_y and left_wrist_y and nose_y:
                if right_wrist_y < nose_y and left_wrist_y < nose_y:
                    cv2.putText(annotated_frame, f'High HANDS UP!!!', (int(width/2 - 125), 30), font, 1, (0, 0, 0), 2)
                elif right_shoulder_y and left_shoulder_y:
                    if right_wrist_y < right_shoulder_y and left_wrist_y < right_shoulder_y:
                        cv2.putText(annotated_frame, f'Low HANDS UP!!!', (int(width/2 - 125), 30), font, 1, (0, 0, 0), 2)
            if right_hip_y and left_hip_y and nose_y:
                    if right_hip_y < nose_y or left_hip_y < nose_y:
                        cv2.putText(annotated_frame, f'MAN DOWN!!!', (int(width/2 - 125), 60), font, 1, (0, 0, 0), 2)

        else:  # If not keypoints are detected then just show the raw frame
            annotated_frame = frame.copy()

        # Write the annotated frame to the output video file
        out.write(annotated_frame)

        # Resize the output frame for display
        annotated_frame = cv2.resize(annotated_frame, (1728, 972))

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release video capture device and close all windows
out.release()
cv2.destroyAllWindows()
