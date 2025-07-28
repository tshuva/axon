import cv2
import imutils
import argparse
import datetime
from multiprocessing import Process, Pipe


def display_times(frames_since_t0, fps):
    local_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # assume that the vid is 30 fps
    video_time = str(datetime.timedelta(seconds=frames_since_t0 / fps))
    return f'Local Time: {local_time}  |  Video Time: {video_time}'


def blur(frame, detections):
    for c in detections:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = frame[y:y+h, x:x+w]
        blurred_roi = cv2.stackBlur(roi, (25, 25))
        frame[y:y+h, x:x+w] = blurred_roi


def normal_display(frame, detections):
    for c in detections:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def displayer(displayer_sub, to_blur, fps):
    action = blur if to_blur else normal_display
    cv2.namedWindow('Detections', cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow('Detections', 800, 600)
    delay = max(1, int(1000 / fps))
    while True:
        frame = displayer_sub.recv()
        if frame is None:
            break

        action(frame["frame"], frame["detections"])

        cv2.putText(frame["frame"], display_times(frames_since_t0=frame['index'], fps=fps), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Detections', frame["frame"])

        if cv2.waitKey(delay) == ord('q'):
            break

    displayer_sub.close()
    cv2.destroyAllWindows()


def decoder(decoder_sub, decoder_pub):
    counter = 0
    prev_frame = None
    while True:
        frame = decoder_sub.recv()
        if frame is None:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == 0:
            prev_frame = gray_frame
            counter += 1
        else:
            diff = cv2.absdiff(gray_frame, prev_frame)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            prev_frame = gray_frame
            counter += 1
            data_to_send = {"frame": frame,
                            "detections": cnts,
                            "index": counter}
            decoder_pub.send(data_to_send)
    decoder_pub.send(None)
    decoder_sub.close()


def video_frame_stream(path):
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        yield frame
    cap.release()


def streamer(path, streamer_pub):
    for frame in video_frame_stream(path):
        streamer_pub.send(frame)
    streamer_pub.send(None)
    streamer_pub.close()


def get_fps(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video FPS: {fps}")
    cap.release()
    return fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video file.')
    parser.add_argument(
        '--path',
        help='Path to the video file',
        default="./People-6387.mp4"
    )

    parser.add_argument('--blur',
                        help='Blur detections', action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    streamer_pub, decoder_sub = Pipe()
    # if it didnt worked i would have made a queue ,and start display after it got a bit full (some arbitrary amount)
    decoder_pub, displayer_sub = Pipe()
    fps = get_fps(args.path)
    p1 = Process(target=streamer, args=(args.path, streamer_pub,))
    p2 = Process(target=decoder, args=(decoder_sub, decoder_pub))
    p3 = Process(target=displayer, args=(displayer_sub, args.blur, fps))

    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
