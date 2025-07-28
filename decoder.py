
import cv2
import imutils


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
