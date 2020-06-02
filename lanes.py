import cv2
import numpy as np
#import matplotlib.pyplot as plt

# Calculate the average line of left lane and right lane
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0] # y-axis
    y2 = 400 # int(y1 * (3/5)) outputs zero
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2]) 

#outputs the averaged lines of left and right
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_avg = np.average(left_fit, axis = 0)
    right_fit_avg = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    return np.array([left_line,right_line])

#outputs canny(edge detection) images 
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0) #smoothens the image
    canny = cv2.Canny(blur,50,150) 
    return canny

#display the lines of left/right lane
def display_lines(image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for x1,y1,x2,y2 in lines:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        return line_image

# identify specific area for finding left/right lane
def region_of_interest(image):
    height = image.shape[0]
    #triangle = np.array((200, height), (1100, height), (550,250))
    polygons = np.array([
    [(200, height), (1100, height), (550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


#test an image
#lanes = cv2.imread("test_image.jpg")
#picture = np.copy(lanes)
#lane_image = canny(picture)   
#cropped_image = region_of_interest(lane_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,np.array([]), 40,5)
#average_lines = average_slope_intercept(picture, lines)
#line_image = display_lines(picture, average_lines)
#combo_image = cv2.addWeighted(picture,0.8,line_image,1.2,1)
#lane_picture = cv2.imshow('result', combo_image)
#cv2.waitKey(0)  

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    lane_image = canny(frame)   
    cropped_image = region_of_interest(lane_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100,np.array([]), 40,5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame,0.8,line_image,1.2,1)
    lane_picture = cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# using matplotlib to find approx coordinates for region of interest
#plt.imshow(lane_image)
#plt.show()