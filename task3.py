import cv2

import math

import numpy as np 

from scipy import signal 

import copy



def edge_detection_sobel(image):

    sobel_operator_x_axis = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.float)
    sobel_operator_y_axis = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.float)

    edge_detection_x_axis = signal.convolve2d(copy.deepcopy(image),sobel_operator_x_axis,'same')
    edge_detection_y_axis = signal.convolve2d(copy.deepcopy(image),sobel_operator_y_axis,'same')

    rows,columns = image.shape
    
    dummy_image_xy = copy.deepcopy(image)

    for i in range(rows):
        for j in range(columns):
            dummy_image_xy[i][j] = math.sqrt(edge_detection_x_axis[i][j] ** 2 + edge_detection_y_axis[i][j] ** 2 )

    return dummy_image_xy

def threshold_operation(image):

    
    threshold_intial = image.max()/4

    a = []

    b = []

    threshold = threshold_seperation(image,a,b,threshold_intial)

    threshold_new = threshold_seperation(image,a,b,threshold)

    for i in range(1000):

        if abs(threshold_new - threshold) >= 0.05:

            threshold = threshold_new

            threshold_new = threshold_seperation(image,a,b,threshold_new)

            print(threshold,threshold_new)

        else:

            break 

    print(threshold) 

    rows,columns = image.shape 

    

    for i in range(rows):

        for j in range(columns):

            if image[i][j] < threshold:

                image[i][j] = 0

    return image

def threshold_seperation(image,a,b,threshold):

     rows, columns = image.shape

     for i in range(rows):

        for j in range(columns):

            if image[i][j] < threshold:

                a.append(image[i][j])

            else:

                b.append(image[i][j]) 
    
    
     new_mean_a = sum(a)/len(a)

     new_mean_b = sum(b)/len(b)

     new_threshold = (new_mean_b + new_mean_a) / 2

     return new_threshold
         
     
     

    

def houghTransform(image):

    rows,columns = image.shape

    accumulator_angle = []


    for i in range(181):
        
        accumulator_angle.append(-90 + i * 1)               

    accumlator_Cells = np.zeros((180,2000))
   
    #print(accumulator_angle)
    
    for i in range(rows):
        
        for j in range(columns):
            
            if image[i][j] != 0:
                
                for k in range(len(accumulator_angle)-1):
                    
                    p = round(i  * math.sin(np.radians(accumulator_angle[k])) + j * math.cos(np.radians(accumulator_angle[k])))
                    
                    m = p + 1000
                    
                    accumlator_Cells[k][m] += 1
                
    return accumlator_Cells

def circle_hough_transform(image):

    rows,columns = image.shape

    accumulator_angle = []


    for i in range(361):
        
        accumulator_angle.append(0 + i * 1)

    accumlator_Cells = np.zeros((600,700))

    for i in range(rows):
        for j in range(columns):
            #for l in range(10):

                if image[i][j] != 0:

                    for k in range(len(accumulator_angle)-1):

                        a = round(i  - (22 + 1) * math.cos(np.radians(accumulator_angle[k])))
                        b = round(j  - (22 + 1) * math.sin(np.radians(accumulator_angle[k])))
                        accumlator_Cells[a,b] += 1
    
   
    return accumlator_Cells


def detect_red_lines_and_blue_lines(accumulator_Cells,image,maximum,minimum,fileName,no_of_lines):

    rows,columns = accumulator_Cells.shape

    max = []

    angle = []

    for i in range(rows):

        for j in range(columns):

                if i > minimum  and i < maximum:

                    max.append(accumulator_Cells[i][j]) 
                    angle.append(i)

    ind = np.argpartition(max, -no_of_lines)[-no_of_lines:]

    for i in range(len(ind)):

        r = ind[i] - 1000
        x1 = int(r * math.cos(np.radians(angle[i] - 90)) - 1000 * math.sin(np.radians(angle[i] - 90)))
        y1 = int(r * math.sin(np.radians(angle[i] - 90)) + 1000 * math.cos(np.radians(angle[i] - 90)))
        x2 = int(r * math.cos(np.radians(angle[i] - 90)) + 1000 * math.sin(np.radians(angle[i] - 90)))
        y2 = int(r * math.sin(np.radians(angle[i] - 90)) - 1000 * math.cos(np.radians(angle[i] - 90)))

        if fileName == 'red_line.jpg':
            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
        else:
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)


    cv2.imwrite(fileName,image)




#%%
if __name__ == "__main__":

    
    gauss_blur_filter = [[0 for x in range(3)] for y in range(3)]

    gauss_blur_filter[0][0] = 1/16 
    gauss_blur_filter[0][1] = 1/8
    gauss_blur_filter[0][2] = 1/16
    gauss_blur_filter[1][0] = 1/8
    gauss_blur_filter[1][1] = 1/4
    gauss_blur_filter[1][2] = 1/8
    gauss_blur_filter[2][0] = 1/16
    gauss_blur_filter[2][1] = 1/8
    gauss_blur_filter[2][2] = 1/16

    image = cv2.imread('hough.jpg')

    image_1 = cv2.imread('hough.jpg')

    image_2 = cv2.imread('hough.jpg')

    grey_image = cv2.imread('hough.jpg',0)

    sharpen_image = signal.convolve2d(grey_image,gauss_blur_filter,'same')

    image_edge_filtered = edge_detection_sobel(copy.deepcopy(sharpen_image))

    threshold_image = threshold_operation(copy.deepcopy(image_edge_filtered))

    cv2.imwrite('threshold.jpg',threshold_image)

    accumulator_Cells = houghTransform(copy.deepcopy(threshold_image))

    detect_red_lines_and_blue_lines(accumulator_Cells,image,91,87,'red_line.jpg',15)

    detect_red_lines_and_blue_lines(accumulator_Cells,image_1,58,53,'blue_lines.jpg',100)
    
    

    accumalator_circle = circle_hough_transform(copy.deepcopy(threshold_image))

    cv2.imwrite('accumulator_circle.jpg',accumalator_circle)



    rows,columns= accumalator_circle.shape

    max_circle = []

    for i in range(rows):

        for j in range(columns):

                    max_circle.append(accumalator_circle[i,j])

    ind2 = np.argpartition(max_circle, -200)[-200:]
    
    print(ind2)

    for i in range(len(ind2)):

        
        r = ind2[i]

        

        b = r % accumalator_circle.shape[1]

        r = r / accumalator_circle.shape[1]

        a = r % accumalator_circle.shape[0]

        cv2.circle(image_2,(int(b),int(a)),int(22),(255,255,0),-1)    

    cv2.imwrite('coin.jpg',image_2)