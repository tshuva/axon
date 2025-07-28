import cv2
import argparse
from multiprocessing import Process, Pipe

from streamer import streamer
from decoder import decoder
from displayer import displayer


def get_fps(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error opening video file")
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
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
    # and do bluring inside the decoder and sample it there (need to find best pracice)
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
