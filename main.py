import argparse

import cv2
import numpy as np


MASK_THRESHOLD = 1
MASK_SUM = 10 ** 5
NUM_CONSECUTIVE = 40



def main():
    parser = argparse.ArgumentParser(description='Get a single frame from each '
                                     'section of a video without movement')
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    capture = cv2.VideoCapture(args.input_path)
    consecutive_count = 0
    _, frame1 = capture.read()
    _, frame2 = capture.read()

    height, width = frame1.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(args.output_path, fourcc, 1, (width, height))

    while frame1 is not None:
        diff = cv2.absdiff(frame1, frame2)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        if np.sum(mask > MASK_THRESHOLD) > MASK_SUM:
            if consecutive_count > NUM_CONSECUTIVE:
                writer.write(frame2)
            consecutive_count = 0
        else:
            consecutive_count += 1

        frame2 = frame1
        _, frame1 = capture.read()

    capture.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
