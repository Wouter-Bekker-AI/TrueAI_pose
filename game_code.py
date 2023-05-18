import cv2
import torch
import random
import time
from ultralytics import YOLO


# Load YOLOv8 model
model = YOLO('yolov8l-pose.pt')

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model.to(device)


def open_video_capture(video_path):
    """
        Open video capture device and return video properties.

        Args:
            video_path (str): Path to the video file.

        Returns:
            cap (cv2.VideoCapture): Video capture object.
            fps (int): Frames per second of the video.
            width (int): Width of the video frame.
            height (int): Height of the video frame.
        """
    # Open video capture device
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps, width, height = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Input has W:{width} H:{height} FPS:{fps}')

    return cap, fps, width, height


def initialize_enemy(width, height):
    """
        Initialize enemy parameters.

        Args:
            width (int): Width of the video frame.
            height (int): Height of the video frame.

        Returns:
            enemy_img (numpy.ndarray): Enemy image array.
            enemy_height (int): Height of the enemy image.
            enemy_width (int): Width of the enemy image.
        """
    # Random position settings
    enemy_img = cv2.imread('enemy.png')  # Load the enemy image
    enemy_img = cv2.resize(enemy_img, (int(width / 6), int(height / 6)))
    enemy_height = int(height / 6)
    enemy_width = int(width / 6)

    return enemy_img, enemy_height, enemy_width


def spawn_enemy(width, height, enemy_width, enemy_height, cap):
    """
        Spawn enemy at a random position.

        Args:
            width (int): Width of the video frame.
            height (int): Height of the video frame.
            enemy_width (int): Width of the enemy image.
            enemy_height (int): Height of the enemy image.
            cap (cv2.VideoCapture): Video capture object.

        Returns:
            enemy_x (int): X-coordinate of the enemy position.
            enemy_y (int): Y-coordinate of the enemy position.
            enemy_start_frame (float): Start frame for enemy visibility.
            enemy_visible (bool): Enemy visibility flag.
        """
    enemy_visible = True

    enemy_x = int(width / 2)
    enemy_y = int(height / 2)
    # The enemy should not spawn too close to the circles
    while 440 > enemy_x > 100 and 290 > enemy_y > 120:
        enemy_x = random.randint(0, width - enemy_width)
        enemy_y = random.randint(0, height - enemy_height)

    # Update the start frame for enemy visibility
    enemy_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    return enemy_x, enemy_y, enemy_start_frame, enemy_visible


def check_start_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y):
    """
        Check if the start touch occurred.

        Args:
            left_palm_x (int): X-coordinate of the left palm.
            left_palm_y (int): Y-coordinate of the left palm.
            right_palm_x (int): X-coordinate of the right palm.
            right_palm_y (int): Y-coordinate of the right palm.

        Returns:
            start_time (float or None): Start time of the game if the start touch occurred, None otherwise.
        """
    global start_touch_start_time, start_text_visible, left_palm_inside, right_palm_inside

    if left_palm_x and left_palm_y:
        if start_text_position[0] < left_palm_x < start_text_position[0] + start_text_size[0] and \
                start_text_position[1] - start_text_size[1] < left_palm_y < start_text_position[1]:
            left_palm_inside = True
        else:
            left_palm_inside = False

    if right_palm_x and right_palm_y:
        if start_text_position[0] < right_palm_x < start_text_position[0] + start_text_size[0] and \
                start_text_position[1] - start_text_size[1] < right_palm_y < start_text_position[1]:
            right_palm_inside = True
        else:
            right_palm_inside = False

    if left_palm_inside or right_palm_inside:
        if start_touch_start_time is None:
            start_touch_start_time = time.time()
            start_time = None
        elif time.time() - start_touch_start_time >= 1:
            start_text_visible = False
            start_time = time.time()  # Start time of the game
        else:
            start_time = None
    else:
        start_touch_start_time = None
        start_time = None

    return start_time


def keypoints(frame):
    """
        Detect keypoints on the frame using the YOLOv8 model.

        Args:
            frame (numpy.ndarray): Input frame.

        Returns:
            left_palm_x (int): X-coordinate of the left palm.
            left_palm_y (int): Y-coordinate of the left palm.
            right_palm_x (int): X-coordinate of the right palm.
            right_palm_y (int): Y-coordinate of the right palm.
            frame (numpy.ndarray): Annotated frame.
        """
    results = model(frame, verbose=False)

    if len(results[0].keypoints) > 0:
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

        # Detect left and right palm positions using hand tracking model
        left_palm_x, left_palm_y = detect_left_palm(left_elbow_y, left_wrist_y, left_wrist_x, left_elbow_x)
        right_palm_x, right_palm_y = detect_right_palm(right_elbow_y, right_wrist_y, right_wrist_x, right_elbow_x)

        cv2.circle(frame, (left_palm_x, left_palm_y), 10, (0, 0, 255), -1)
        cv2.circle(frame, (right_palm_x, right_palm_y), 10, (0, 0, 255), -1)

        return left_palm_x, left_palm_y, right_palm_x, right_palm_y, frame
    else:
        return None, None, None, None, frame


def draw_on_frame(frame, enemy_x=None, enemy_y=None, enemy_width=None, enemy_height=None, enemy_img=None, width=640, height=480):
    """
        Draw annotations on the frame.

        Args:
            frame (numpy.ndarray): Input frame.
            enemy_x (int): X-coordinate of the enemy position.
            enemy_y (int): Y-coordinate of the enemy position.
            enemy_width (int): Width of the enemy image.
            enemy_height (int): Height of the enemy image.
            enemy_img (numpy.ndarray): Enemy image array.
            width (int): Width of the video frame.
            height (int): Height of the video frame.

        Returns:
            frame (numpy.ndarray): Annotated frame.
        """
    global start_text_visible, start_text_position, font, start_text_size, score, time_left
    if start_text_visible:
        cv2.rectangle(frame, (start_text_position[0] - 5, start_text_position[1] + 5), (start_text_position[0] + start_text_size[0] + 5, start_text_position[1] - start_text_size[1] - 5), (0, 0, 0), 3)
        cv2.putText(frame, 'START', start_text_position, font, 2, (0, 255, 0), 2)
    if enemy_x is not None and enemy_y is not None and enemy_width is not None and enemy_height is not None and enemy_img is not None:
        # Overlay enemy image on the annotated frame
        frame[enemy_y:enemy_y + enemy_height, enemy_x:enemy_x + enemy_width] = enemy_img

    if time_left:
        # Display the timer in the top right corner
        timer_text = f'Time: {int(time_left)}s'
        cv2.putText(frame, timer_text, (width - 200, 30), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Score:{score}', (10, 30), font, 1, (0, 0, 255), 2)

    return frame


def read_frame(cap):
    """
        Read a frame from the video capture.

        Args:
            cap (cv2.VideoCapture): Video capture object.

        Returns:
            tuple: A tuple containing the boolean indicating if the frame was read successfully and the video frame.

        """
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    return ret, frame


def detect_left_palm(left_elbow_y, left_wrist_y, left_wrist_x, left_elbow_x):
    """
        Perform left palm detection and return its coordinates.

        Args:
            left_elbow_y (float): y-coordinate of the left elbow.
            left_wrist_y (float): y-coordinate of the left wrist.
            left_wrist_x (float): x-coordinate of the left wrist.
            left_elbow_x (float): x-coordinate of the left elbow.

        Returns:
            tuple: A tuple containing the x and y coordinates of the left palm.

        """
    # Perform left palm detection and return its position
    if left_elbow_y and left_wrist_y:
        left_palm_x = int(0.35 * (left_wrist_x - left_elbow_x) + left_wrist_x)
        left_palm_y = int(0.35 * (left_wrist_y - left_elbow_y) + left_wrist_y)
    else:
        left_palm_x, left_palm_y = (None, None)
    return left_palm_x, left_palm_y


def detect_right_palm(right_elbow_y, right_wrist_y, right_wrist_x, right_elbow_x):
    """
        Perform right palm detection and return its coordinates.

        Args:
            right_elbow_y (float): y-coordinate of the right elbow.
            right_wrist_y (float): y-coordinate of the right wrist.
            right_wrist_x (float): x-coordinate of the right wrist.
            right_elbow_x (float): x-coordinate of the right elbow.

        Returns:
            tuple: A tuple containing the x and y coordinates of the right palm.

        """
    # Perform right palm detection and return its position
    if right_wrist_y and right_elbow_y:
        right_palm_x = int(0.35 * (right_wrist_x - right_elbow_x) + right_wrist_x)
        right_palm_y = int(0.35 * (right_wrist_y - right_elbow_y) + right_wrist_y)
    else:
        right_palm_x, right_palm_y = (None, None)
    return right_palm_x, right_palm_y


def check_enemy_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y, enemy_x, enemy_y, enemy_width, enemy_height):
    """
        Check if the palms touch the enemy image and update the score accordingly.

        Args:
            left_palm_x (int): x-coordinate of the left palm.
            left_palm_y (int): y-coordinate of the left palm.
            right_palm_x (int): x-coordinate of the right palm.
            right_palm_y (int): y-coordinate of the right palm.
            enemy_x (int): x-coordinate of the enemy.
            enemy_y (int): y-coordinate of the enemy.
            enemy_width (int): Width of the enemy.
            enemy_height (int): Height of the enemy.

        Returns:
            bool: True if the enemy is touched, False otherwise.

        """
    global score
    enemy_visible = True
    # Check if the palms touch the enemy image
    if left_palm_x and left_palm_y:
        if enemy_x <= left_palm_x <= enemy_x + enemy_width and enemy_y <= left_palm_y <= enemy_y + enemy_height:
            enemy_visible = False
            score += 1

    if right_palm_x and right_palm_y:
        if enemy_x <= right_palm_x <= enemy_x + enemy_width and enemy_y <= right_palm_y <= enemy_y + enemy_height:
            enemy_visible = False
            score += 1
    return enemy_visible


def check_inside_circles(width, height, left_palm_x, left_palm_y, right_palm_x, right_palm_y, frame):
    """
        Check if the palm coordinates are inside the circles and update the palm colors accordingly.

        Args:
            width (int): Width of the frame.
            height (int): Height of the frame.
            left_palm_x (int): x-coordinate of the left palm.
            left_palm_y (int): y-coordinate of the left palm.
            right_palm_x (int): x-coordinate of the right palm.
            right_palm_y (int): y-coordinate of the right palm.
            frame: Frame to draw the circles on.

        Returns:
            tuple: A tuple containing a boolean value indicating if both palms are inside the circles and the annotated frame.

        """
    global left_palm_inside, right_palm_inside
    # Circles
    circle_size = 45
    circle_1_x = int(0.4 * width)
    circle_2_x = int(0.6 * width)
    circle_y = int(0.5 * height)

    # Check if palm is inside the circles, if the palm is inside the circle then circle is green else yellow.
    if left_palm_x and left_palm_y:
        if abs(left_palm_x - circle_1_x) <= circle_size and abs(left_palm_y - circle_y) <= circle_size:
            left_palm_inside = True
            cv2.circle(frame, (circle_1_x, circle_y), circle_size, (0, 255, 0), 2)
        else:
            left_palm_inside = False
            cv2.circle(frame, (circle_1_x, circle_y), circle_size, (0, 255, 255), 2)
    if right_palm_x and right_palm_y:
        if abs(right_palm_x - circle_2_x) <= circle_size and abs(right_palm_y - circle_y) <= circle_size:
            right_palm_inside = True
            cv2.circle(frame, (circle_2_x, circle_y), circle_size, (0, 255, 0), 2)
        else:
            right_palm_inside = False
            cv2.circle(frame, (circle_2_x, circle_y), circle_size, (0, 255, 255), 2)
    if not left_palm_x:
        cv2.circle(frame, (circle_1_x, circle_y), circle_size, (0, 255, 255), 2)
    if not right_palm_x:
        cv2.circle(frame, (circle_2_x, circle_y), circle_size, (0, 255, 255), 2)
    if left_palm_inside and right_palm_inside:
        return True, frame
    else:
        return False, frame


def output(frame, out):
    """
        Write the annotated frame to the output video file and display it.

        Args:
            frame: Annotated frame.
            out: Video writer object.

        """
    # Write the annotated frame to the output video file
    out.write(frame)

    # Resize the output frame for display
    frame = cv2.resize(frame, (1728, 972))

    cv2.imshow('Game', frame)


def main(video_path):
    """
        Main function to run the game.

        Args:
            video_path (str): Path to the video file or camera index.

        """
    global left_palm_inside, right_palm_inside, time_left, score, start_text_visible
    # Open video capture device
    cap, fps, width, height = open_video_capture(video_path)

    # Define video writer
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    # Initialize enemy parameters
    enemy_img, enemy_height, enemy_width = initialize_enemy(width, height)

    # Game settings
    game_duration = 30  # in seconds
    start_time = None
    show_enemy = False

    # Start the game loop
    while cap.isOpened():
        while start_time is None:
            ret, raw_frame = read_frame(cap)

            left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame = keypoints(raw_frame)

            # Check if start touch occurred
            start_time = check_start_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y)

            annotated_frame = draw_on_frame(annotated_frame)

            output(annotated_frame, out)

            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        while time.time() - start_time < game_duration:
            time_left = game_duration - int(time.time() - start_time)
            if time_left <= 0:
                break
            left_palm_inside = False
            right_palm_inside = False
            while not show_enemy:
                time_left = game_duration - int(time.time() - start_time)
                ret, raw_frame = read_frame(cap)
                left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame = keypoints(raw_frame)
                both_palms_in, annotated_frame = check_inside_circles(width, height, left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame)

                annotated_frame = draw_on_frame(annotated_frame)

                output(annotated_frame, out)

                if both_palms_in:
                    show_enemy = True

                # Check for key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

            if time_left <= 0:
                break

            enemy_x, enemy_y, enemy_start_frame, show_enemy = spawn_enemy(width, height, enemy_width, enemy_height, cap)

            left_palm_inside = False
            right_palm_inside = False

            while show_enemy:
                time_left = game_duration - int(time.time() - start_time)
                if time_left <= 0:
                    break
                ret, raw_frame = read_frame(cap)
                left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame = keypoints(raw_frame)
                both_palms_in, annotated_frame = check_inside_circles(width, height, left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame)
                annotated_frame = draw_on_frame(annotated_frame, enemy_x, enemy_y, enemy_width, enemy_height, enemy_img)
                show_enemy = check_enemy_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y, enemy_x, enemy_y, enemy_width, enemy_height)

                output(annotated_frame, out)

                # Check for key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

        frame_count = 0
        while frame_count < 75:
            ret, raw_frame = read_frame(cap)
            # Display the final score in the center of the screen
            cv2.putText(raw_frame, f'Final Score: {score}', (100, int(height / 2)), font,
                        2, (0, 0, 255), 5)

            output(raw_frame, out)

            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

            frame_count += 1

        start_time = None
        score = 0
        start_text_visible = True

    # Release the video capture device and close the window
    cap.release()
    cv2.destroyAllWindows()


# Entry point of the program
if __name__ == '__main__':
    # Global variables
    video_path = 0
    start_text_position = (228, 219)
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_text_size = (184, 43)
    start_touch_start_time = None
    start_text_visible = True
    left_palm_inside = False
    right_palm_inside = False
    score = 0
    time_left = None

    main(video_path)
