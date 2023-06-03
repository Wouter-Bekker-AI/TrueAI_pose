import os.path
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


def open_video_capture(video_path, webcam):
    # Open video capture device
    if webcam:
        '''cap = cv2.VideoCapture(video_path)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, codec)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)'''

        cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        #codec = 0x47504A4D  # MJPG
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, codec)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    else:
        cap = cv2.VideoCapture(video_path)

    # Get video properties
    print(f'Input has W:{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} H:{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} FPS:{int(cap.get(cv2.CAP_PROP_FPS))}')
    fps, width, height = 30, 640, 480

    return cap, fps, width, height


def initialize_enemy(width, height):
    # Random position settings
    enemy_img = cv2.imread('enemy.png')  # Load the enemy image
    enemy_img = cv2.resize(enemy_img, (int(width / 6), int(height / 6)))
    enemy_height = int(height / 6)
    enemy_width = int(width / 6)

    return enemy_img, enemy_height, enemy_width


def spawn_enemy(enemy_width, enemy_height, cap):
    enemy_visible = True

    enemy_x = int(web_width / 2)
    enemy_y = int(web_height / 2)
    # The enemy should not spawn too close to the circles
    while 440 > enemy_x > 100 and 290 > enemy_y > 120:
        enemy_x = random.randint(0, web_width - enemy_width)
        enemy_y = random.randint(0, web_height - enemy_height)

    # Update the start frame for enemy visibility
    enemy_start_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    return enemy_x, enemy_y, enemy_start_frame, enemy_visible


def keypoints(frame, presenting=False):
    hands_up = False
    left_palm_x, left_palm_y, right_palm_x, right_palm_y = None, None, None, None
    hh, lh, md = False, False, False

    results = model(frame, verbose=False)

    if len(results[0].keypoints) > 0:

        if presenting:
            frame = results[0].plot(img=frame)

        for i in range(len(results[0].keypoints)):
            # Key the locations of each joint
            keypoints = results[0].keypoints[i].squeeze().tolist()

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

            if left_palm_x and left_palm_y:
                cv2.circle(frame, (left_palm_x, left_palm_y), 10, (0, 0, 255), -1)
            if right_palm_x and right_palm_y:
                cv2.circle(frame, (right_palm_x, right_palm_y), 10, (0, 0, 255), -1)

            # If back is towards cam then infer that nose_y is same as ear_y
            if not nose_y:
                if right_ear_y or left_ear_y:
                    if right_ear_y:
                        nose_y = right_ear_y
                    elif left_ear_y:
                        nose_y = left_ear_y

            # Check if hands are up
            if right_wrist_y and left_wrist_y and nose_y:
                if right_wrist_y < nose_y and left_wrist_y < nose_y:
                    hands_up = True
                    hh = True
                elif right_shoulder_y and left_shoulder_y:
                    if right_wrist_y < right_shoulder_y and left_wrist_y < right_shoulder_y:
                        hands_up = True
                        lh = True
            if right_hip_y and left_hip_y and nose_y:
                if right_hip_y < nose_y or left_hip_y < nose_y:
                    md = True

            if not presenting:
                return left_palm_x, left_palm_y, right_palm_x, right_palm_y, frame, hands_up

        if presenting:
            if hh:
                cv2.putText(frame, f'High HANDS UP!!!', (int(web_width / 2 - 175), 30), font, 1, (0, 0, 0), 2)
                if not frame_count % 3 == 0:
                    cv2.putText(frame, f'Silent Alarm Triggered!', (int(web_width / 2 - 225), 60), font, 1, (50, 255, 255), 2)
            elif lh:
                cv2.putText(frame, f'Low HANDS UP!!!', (int(web_width / 2 - 175), 30), font, 1, (0, 0, 0), 2)
                if not frame_count % 3 == 0:
                    cv2.putText(frame, f'Silent Alarm Triggered!', (int(web_width / 2 - 225), 60), font, 1, (50, 255, 255), 2)
            elif md:
                cv2.putText(frame, f'MAN DOWN!!!', (int(web_width / 2 - 175), 30), font, 1, (0, 0, 0), 2)

        return left_palm_x, left_palm_y, right_palm_x, right_palm_y, frame, hands_up
    else:
        return None, None, None, None, frame, hands_up


def draw_on_frame(frame, enemy_x=None, enemy_y=None, enemy_width=None, enemy_height=None, enemy_img=None, width=640, height=480):
    global font, score, time_left, high_score

    if enemy_x is not None and enemy_y is not None and enemy_width is not None and enemy_height is not None and enemy_img is not None:
        # Overlay enemy image on the annotated frame
        frame[enemy_y:enemy_y + enemy_height, enemy_x:enemy_x + enemy_width] = enemy_img

    if time_left:
        # Display the timer in the top right corner
        timer_text = f'Time: {int(time_left)}s'
        cv2.putText(frame, timer_text, (width - 200, 30), font, 1, (0, 0, int(255 - (time_left*8.5)) ), 2)
        cv2.putText(frame, f'Score:{score}', (250, 30), font, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'High Score:{high_score}', (10, 30), font, 1, (255, 0, 0), 2)

    return frame


def read_frame(cap, flip=True):
    global frame_count
    frame_count += 1
    ret, frame = cap.read()
    if flip:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
    return ret, frame


def detect_left_palm(left_elbow_y, left_wrist_y, left_wrist_x, left_elbow_x):
    # Perform left palm detection and return its position
    if left_elbow_y and left_wrist_y:
        left_palm_x = int(0.35 * (left_wrist_x - left_elbow_x) + left_wrist_x)
        left_palm_y = int(0.35 * (left_wrist_y - left_elbow_y) + left_wrist_y)
    else:
        left_palm_x, left_palm_y = (None, None)
    return left_palm_x, left_palm_y


def detect_right_palm(right_elbow_y, right_wrist_y, right_wrist_x, right_elbow_x):
    # Perform right palm detection and return its position
    if right_wrist_y and right_elbow_y:
        right_palm_x = int(0.35 * (right_wrist_x - right_elbow_x) + right_wrist_x)
        right_palm_y = int(0.35 * (right_wrist_y - right_elbow_y) + right_wrist_y)
    else:
        right_palm_x, right_palm_y = (None, None)
    return right_palm_x, right_palm_y


def check_enemy_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y, enemy_x, enemy_y, enemy_width, enemy_height):
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


def check_inside_circles(left_palm_x, left_palm_y, right_palm_x, right_palm_y, frame):
    global left_palm_inside, right_palm_inside
    # Circles
    circle_size = 45
    circle_1_x = int(0.4 * web_width)
    circle_2_x = int(0.6 * web_width)
    circle_y = int(0.5 * web_height)

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


def output(frame):
    # Resize the output frame for display
    frame = cv2.resize(frame, (1920, 1080))

    # Write the annotated frame to the output video file
    video_writer.write(frame)

    cv2.imshow('Watcher', frame)


def main_menu():
    global web_cap, video_cap

    if video_cap:
        while True:
            ret, raw_frame = read_frame(web_cap)
            ret_2, raw_frame_2 = read_frame(video_cap, flip=False)
            if not ret_2:
                video_cap.release()
                video_cap, fps_2, width_2, height_2 = open_video_capture(video_path, webcam=False)
                ret_2, raw_frame_2 = read_frame(video_cap, flip=False)

            left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame)
            if hands_up:
                break
            else:
                output(raw_frame_2)

                # Check for key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

    while True:
        ret, raw_frame = read_frame(web_cap)
        if not ret:
            web_cap.release()
            web_cap, web_fps, web_width, web_height = open_video_capture(video_path, webcam=False)
            ret, raw_frame = read_frame(web_cap, flip=False)

        left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame)

        present = present_button.check_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y)
        game = game_button.check_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y)

        annotated_frame = present_button.render(raw_frame)
        annotated_frame = game_button.render(annotated_frame)

        output(annotated_frame)

        if present or game:
            if present:
                presentation()
            elif game:
                play_game()

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break


def presentation():
    while True:
        ret, raw_frame = read_frame(web_cap)

        left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame, presenting=True)

        if back_button.check_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y):
            main_menu()

        annotated_frame = back_button.render(annotated_frame)

        output(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def play_game():
    global score, high_score, time_left
    # Initialize enemy parameters
    enemy_img, enemy_height, enemy_width = initialize_enemy(web_width, web_height)

    # Game settings
    game_duration = 30  # in seconds
    start_time = None
    show_enemy = False
    score = 0

    while start_time is None:
        ret, raw_frame = read_frame(web_cap)

        left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame)

        if back_button.check_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y):
            main_menu()

        # Check if start touch occurred
        start = start_button.check_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y)
        if start:
            start_time = time.time()

        annotated_frame = start_button.render(raw_frame)
        annotated_frame = back_button.render(annotated_frame)

        annotated_frame = draw_on_frame(annotated_frame)

        output(annotated_frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    while time.time() - start_time < game_duration:
        time_left = game_duration - int(time.time() - start_time)
        if time_left <= 0:
            break

        while not show_enemy:
            time_left = game_duration - int(time.time() - start_time)
            if time_left <= 0:
                break

            ret, raw_frame = read_frame(web_cap)
            left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame)
            both_palms_in, annotated_frame = check_inside_circles(left_palm_x, left_palm_y, right_palm_x,
                                                                  right_palm_y, annotated_frame)

            annotated_frame = draw_on_frame(annotated_frame)

            output(annotated_frame)

            if both_palms_in:
                show_enemy = True

            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        enemy_x, enemy_y, enemy_start_frame, show_enemy = spawn_enemy(enemy_width, enemy_height, web_cap)

        while show_enemy:
            time_left = game_duration - int(time.time() - start_time)
            if time_left <= 0:
                break
            ret, raw_frame = read_frame(web_cap)
            left_palm_x, left_palm_y, right_palm_x, right_palm_y, annotated_frame, hands_up = keypoints(raw_frame)
            both_palms_in, annotated_frame = check_inside_circles(left_palm_x, left_palm_y, right_palm_x,
                                                                  right_palm_y, annotated_frame)
            annotated_frame = draw_on_frame(annotated_frame, enemy_x, enemy_y, enemy_width, enemy_height, enemy_img)
            show_enemy = check_enemy_touch(left_palm_x, left_palm_y, right_palm_x, right_palm_y, enemy_x, enemy_y,
                                           enemy_width, enemy_height)

            output(annotated_frame)

            # Check for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    f_count = 0
    while f_count < 75:
        ret, raw_frame = read_frame(web_cap)
        # Display the final score in the center of the screen
        if score > high_score:
            cv2.putText(raw_frame, f'New High Score:{score}', (10, int(web_height / 2)), font,
                        2, (0, 0, 255), 4)
        else:
            cv2.putText(raw_frame, f'Final Score: {score}', (100, int(web_height / 2)), font,
                        2, (0, 0, 255), 5)

        output(raw_frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        f_count += 1

    if score > high_score:
        high_score = score

    main_menu()


class Button:
    def __init__(self, position, text):
        self.text = text
        self.position = position
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.size = cv2.getTextSize(self.text, self.font, 2, 2)[0]
        self.inside = False
        self.touch_start_time = None
        self.activated = False
        self.right_palm_inside = False
        self.left_palm_inside = False

    def render(self, frame):
        if self.touch_start_time:
            for i in range(round((time.time() - self.touch_start_time) / 0.2)):
                cv2.line(frame, (self.position[0] - 5, self.position[1] + 5 - (5 * i)),
                         (self.position[0] + self.size[0] + 5, self.position[1] + 5 - (5 * i)),
                         (0, 255, 0), 4)
        cv2.rectangle(frame, (self.position[0] - 5, self.position[1] + 5), (self.position[0] + self.size[0] + 5, self.position[1] - self.size[1] - 5), (0, 255, 0), 3)
        cv2.putText(frame, self.text, self.position, self.font, 2, (185, 25, 25), 2)
        return frame

    def check_touch(self, left_palm_x, left_palm_y, right_palm_x, right_palm_y):
        self.activated = False
        if left_palm_x and left_palm_y:
            if self.position[0] < left_palm_x < self.position[0] + self.size[0] and \
                    self.position[1] - self.size[1] < left_palm_y < self.position[1]:
                self.left_palm_inside = True
            else:
                self.left_palm_inside = False
        else:
            self.left_palm_inside = False

        if right_palm_x and right_palm_y:
            if self.position[0] < right_palm_x < self.position[0] + self.size[0] and \
                    self.position[1] - self.size[1] < right_palm_y < self.position[1]:
                self.right_palm_inside = True
            else:
                self.right_palm_inside = False
        else:
            self.right_palm_inside = False

        if self.left_palm_inside or self.right_palm_inside:
            if self.touch_start_time is None:
                self.touch_start_time = time.time()
            elif time.time() - self.touch_start_time >= 2:
                self.touch_start_time = None
                self.activated = True
                return self.activated
        else:
            self.touch_start_time = None
            return self.activated


# Entry point of the program
if __name__ == '__main__':
    # Global variables
    webcam_path = 0
    video_path = f'Exhibition_trial_0.mp4'
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0

    start_button = Button((228, 219), 'START')
    game_button = Button((228, 319), 'Game')
    present_button = Button((180, 169), 'PRESENT')
    back_button = Button((465, 50), 'BACK')

    left_palm_inside = False
    right_palm_inside = False
    score = 0
    high_score = 2
    time_left = None

    # Define video writer
    video_writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1920, 1080))

    # Open video capture device
    web_cap, web_fps, web_width, web_height = open_video_capture(webcam_path, webcam=True)
    if os.path.isfile(f'Exhibition_trial_0.mp4'):
        video_cap, video_fps, video_width, video_height = open_video_capture(video_path, webcam=False)
    else:
        video_cap = None

    main_menu()
