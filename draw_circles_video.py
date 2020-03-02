#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Matheus Dib, Fabio de Miranda"

#importações do Vídeo
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
#Importações para a Máscara
from ipywidgets import widgets, interact, IntSlider
import auxiliar as aux
import math

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
#cap = cv2.VideoCapture('hall_box_battery.mp4')

# Parameters to use when opening the webcam.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1
counter = 0
variavel = 0
circle1 = np.array([0,0])
circle2 = np.array([0,240])
angulo = 0
h = 0
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

print("Press q to QUIT")

# Returns an image containing the borders of the image
# sigma is how far from the median we are setting the thresholds
def auto_canny(image, sigma=0.33):
    """compute the median of the single channel pixel intensities"""
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame_hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##hsv's do Ciano
    hsv1_ciano, hsv2_ciano = aux.ranges('#004586')
    
    ##hsv's de Magenta
    hsv1_magenta, hsv2_magenta = aux.ranges('#88123F')
    
    ##Mask do Ciano
    mask_ciano = cv2.inRange(frame_hsv, hsv1_ciano, hsv2_ciano)
    
    ##Mask da Magenta
    mask_magenta = cv2.inRange(frame_hsv, hsv1_magenta, hsv2_magenta)
    
    ##Mascára Final
    mask = mask_ciano + mask_magenta
    
    ##Seleção
    # selecao = cv2.bitwise_and(frame, frame, mask=mask)
    segmentado = cv2.morphologyEx(mask,cv2.MORPH_CLOSE, np.ones((5, 5)))
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_RECT, np.ones((5, 5)))
    selecao = cv2.bitwise_and(frame, frame, mask=segmentado)

    
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #blur = gray
    
    # Detect the edges present in the image
    bordas = auto_canny(blur)

    # circles = []

    # Obtains a version of the edges image where we can draw in color
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    
    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    circles = None
    circles = cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    if circles is not None:        
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # print(i)
            # draw the outer circle
          # cv2.circle(img,        center,  radius, color[, thickness[, lineType[, shift]]])
            cv2.circle(selecao, (i[0],i[1]), i[2], (0,255,0), 2)
            # draw the center of the circle
            cv2.circle(selecao, (i[0],i[1]), 2, (0,0,255), 3)

                
    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(bordas_color, (0,0), (511,511), (255,0,0), 5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(384,0),(510,128),(0,255,0),3)
    counter += 1
    # print(counter)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    try: 
        cv2.line(selecao, (circles[0][0][0], circles[0][0][1]), (circles[0][1][0], circles[0][1][1]), (0,255,0), 2)
        
        # if counter == 20:
        #     counter = 0
        # x1 = circles[0][0][0]
        # x2 = circles[0][1][0]
        # y1 = circles[0][0][1]
        # y2 = circles[0][1][1]

        circle1 = np.array(circles[0][0][:2], dtype='int64') # converter para int64 antes de realizar operações
        circle2 = np.array(circles[0][1][:2], dtype='int64')
        print("circle1 = ", circle1)
        print("circle2 = ", circle2)
        vetor1_2 = (circle1-circle2)
        print("vetor1_2:",vetor1_2)
        tangente = -vetor1_2[1]/vetor1_2[0]
        angulo = math.degrees( math.atan(tangente) )
        print("angulo: %.0f°" %angulo)
        f = 444 #17.0
        H = 13.8
        h = math.sqrt( vetor1_2.dot( vetor1_2 ) ) # forma otimizada de calcular distancia
        print("h = %.2f" %h)
        variavel = round( (H * f / h), 2)
        print("A distância é ", variavel,"\n")
        
    except:
        pass
    
    center = np.array(((circle1+circle2)/2),dtype='int64')
    # adicionar textos na tela:

    cv2.putText(selecao," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(selecao,"  h="+str(h)+" pixels",(0,100), font2, 1.2,(255,255,255), 2, cv2.LINE_AA)
    # cv2.putText(selecao,"distancia = "+str(variavel)+" cm", tuple(center) , font2, 1.2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(selecao,"  distancia = "+str(variavel)+" cm",(0,120), font2, 1.2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(selecao,"  angulo = %.0f degrees" %angulo,(0,140), font2, 1.2, (255,255,255), 2, cv2.LINE_AA)




    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('Detector de circulos',selecao)
    # cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
