################################################################
# Author: Ronell Bresler
# Module: LineDetect.py
#
#
# References:
# https://www.analyticsvidhya.com/blog/2020/05/tutorial-real-time-lane-detection-opencv/
# https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132
# https://medium.com/computer-car/udacity-self-driving-car-nanodegree-project-1-finding-lane-lines-9cd6a846c58c
# https://campushippo.com/lessons/detect-highway-lane-lines-with-opencv-and-python-21438a3e2
################################################################

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Inputfile:
    def __init__(self, inputimage, height, width, loaded_img):
        self.inputimage = inputimage
        self.height = height
        self.width = width
        self.loaded_img = cv2.imread(inputimage)


def main():

    inputfile = Inputfile("SampleIMG/paint.png", 0, 0, 0)
    inputfile.height = inputfile.loaded_img.shape[0]
    inputfile.width = inputfile.loaded_img.shape[1]

    plt.figure()
    plt.imshow(cv2.cvtColor(inputfile.loaded_img, cv2.COLOR_BGR2RGB))

    region_of_interest_vertices = Set_roi_vertices(inputfile.height, inputfile.width)

    # Canny filter
    canny_edges = Canny_edge_detector(inputfile.loaded_img)
    
    # Crop img with roi
    cropped_image = Region_of_interest(canny_edges, np.array([region_of_interest_vertices], np.int32), inputfile.height, inputfile.width)

    Draw_Hough(cropped_image, inputfile)


################################################################
def Draw_Hough(cropped_image, inputfile):
    lines = cv2.HoughLinesP(cropped_image,
                    rho=6,
                    theta=np.pi/180,
                    threshold=50,
                    lines=np.array([]),
                    minLineLength=40,
                    maxLineGap=100)

    plt.imshow(Draw_lines(inputfile.loaded_img, lines))
    plt.show()
    return 

################################################################
def Canny_edge_detector(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(gray, 100, 200)
    Plot_show_BGR(canny_image)
    
    return canny_image  

################################################################
def Region_of_interest(img, vertices, height, width):

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    Plot_show_BGR(masked_image)

    return masked_image

################################################################
def Draw_lines(img, lines):
       
    color = [0, 255, 0] # green
    thickness = 5

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1,y1), (x2,y2), color, thickness)

    return img

################################################################
def Set_roi_vertices(height, width):

    region_of_interest_vertices = [
    (0, height),
    (round(width/1.9), round(height/1.9)),
    (width, height)
    ]

    return region_of_interest_vertices


################################################################
def Plot_show_BGR(method):
    plt.figure()
    plt.imshow(cv2.cvtColor(method, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()