################################################################
# Author: Ronell Bresler
# Module: VideoLineDetect.py
#
#
# References:
# https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
# https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
# https://medium.com/computer-car/udacity-self-driving-car-nanodegree-project-1-finding-lane-lines-9cd6a846c58c
# https://campushippo.com/lessons/detect-highway-lane-lines-with-opencv-and-python-21438a3e2
# https://www.youtube.com/watch?v=G0cHyaP9HaQ
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
################################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Inputfile:
    def __init__(self, cap, height, width, frame):
        self.cap = cap
        self.height = height
        self.width = width
        self.frame = frame

def main():

    inputfile = Inputfile(cv2.VideoCapture('SampleIMG/gmod2.mp4'), 0, 0, 0)

    while inputfile.cap.isOpened():
        ret, frame = inputfile.cap.read()
        inputfile.frame = frame
        inputfile.height = inputfile.frame.shape[0]
        inputfile.width = inputfile.frame.shape[1]

        frame1 = One_frame(inputfile)
        cv2.imshow('frame', frame1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

################################################################
def One_frame(inputfile):

    region_of_interest_vertices = Set_region_of_interest_vertices(inputfile.height, inputfile.width)

    # Canny filter
    canny_edges = Canny_edge_detector(inputfile.frame)

    # Crop img with roi
    cropped_image = Region_of_interest(canny_edges, np.array([region_of_interest_vertices], np.int32), inputfile.height, inputfile.width)

    lines = cv2.HoughLinesP(cropped_image,
                        rho=6,
                        theta=np.pi/180,
                        threshold=160,
                        lines=np.array([]),
                        minLineLength=40,
                        maxLineGap=25)

    return Draw_lines(inputfile.frame, lines)

################################################################
def Canny_edge_detector(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(gray, 100, 200)
    
    return canny_image  

################################################################
def Region_of_interest(img, vertices, height, width):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

################################################################
def Draw_lines(img, lines):
       
    color = [0, 255, 0] # green
    thickness = 10

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)

    return img

################################################################
def Set_region_of_interest_vertices(height, width):
    
    region_of_interest_vertices = [
    (0, height),
    (round(width/1.9), round(height/1.9)),
    (width, height)
    ]

    return region_of_interest_vertices


if __name__ == "__main__":
    main()