import time
import cv2
import datetime


def display_times(frames_since_t0, fps):
    local_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    video_time = str(datetime.timedelta(seconds=frames_since_t0 / fps))
    return f'Local Time: {local_time}  |  Video Time: {video_time}'


def blur(frame, detections):
    for c in detections:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.stackBlur(roi, (15, 15))
        frame[y:y+h, x:x+w] = blurred_roi


def normal_display(frame, detections):
    for c in detections:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def displayer(displayer_sub, to_blur, fps):
    action = blur if to_blur else normal_display
    cv2.namedWindow('Detections', cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('Detections', 800, 600)
    frame_interval_ms = 1000 / fps
    while True:
        frame = displayer_sub.recv()
        start_time = time.time()
        if frame is None:
            break

        action(frame["frame"], frame["detections"])

        cv2.putText(frame["frame"], display_times(frames_since_t0=frame['index'], fps=fps), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detections', frame["frame"])

        elapsed_ms = (time.time() - start_time) * 1000
        delay = max(1, int(frame_interval_ms - elapsed_ms))
        if cv2.waitKey(delay) == ord('q'):
            break

    displayer_sub.close()
    cv2.destroyAllWindows()
