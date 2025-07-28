import cv2


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
