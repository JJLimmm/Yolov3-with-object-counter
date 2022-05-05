########################################################################################################################
# do all the necessary imports
#Dependencies#

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet

########################################################################################
#---------------------------YOLO MODEL INITIALISATION----------------------------------#
#If there is more than one model, copypaste below and change params names

#--------------------------------MAIN MODEL--------------------------------------------#
configfile = "/content/darknet/cfg/yolov3_test.cfg"
datafile = "/content/darknet/data/obj_JJ.data"
weightsfile = "/content/gdrive/MyDrive/yolov3/backup/yolov3_custom2_final.weights"

network, class_names, class_colors = darknet.load_network(configfile,datafile,weightsfile,batch_size=1)
width = darknet.network_width(network)
height = darknet.network_height(network)
#--------------------------------------------------------------------------------------#

##################################################################################################

#----------------------------FOR LINKING VIDEO STREAM----------------------------------#

#getting the video stream from IP camera
#cap = cv2.VideoCapture("rtsp://10.7.5.105/1/h264major")              #Linking openCV to IP camera via 'cap' variable
cap = cv2.VideoCapture("r_test_2.avi")                             #if detecting from offline video
print("Video Ready!")

vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))                            #set camera frame width as 'width'
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))                           #set camera frame height as 'height'
print(vid_width,vid_height)                                                        #display camera frame dimensions

#out = cv2.VideoWriter("invt_detections.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 20.0,(800,600))     #cv2.VideoWriter("videofilename to save as" , "video file type to save as in fourCC codec" , fps , frame dimensions)
#print(image_width,image_height)
#---------------------------------------------------------------------------------------#
########################################################################################

#----------------------YOLO Detections and Invt List creation---------------------------#

#Initialising counter for recording detected objects
windows_installed_count = 0
windows_uninstalled_count = 0
doors_uninstalled_count = 0
doors_installed_count = 0
electrical_power_count = 0
electrical_switch_count = 0
electrical_telecom_count = 0
electrical_lights_count = 0
electrical_uninstalled_count = 0
electrical_mains_count = 0
pvc_pipes_count = 0
cement_bag_count = 0
exit_signage_count = 0
wires_count = 0

while True: 	
    #Continuously run on a loop until video feed is stopped
    #Detections will continue to run and output 
        
    ret,frame = cap.read()                                                  #cap.read() returns a bool (True / False). If frame is read correctly, it will be True.

    #ret, frame_read = cap.read() # read image frame
    if ret==False: break
 
    #prep video frames to input into YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                                  #convert colour from openCV (frame) to YOLO format (frame_rgb)
    frame_rgb_resized = cv2.resize(frame_rgb, (width,height), interpolation=cv2.INTER_LINEAR)           #resize color_img to YOLO input dimensions (416 x 416) can be checked from .cfg file
    img_for_detect = darknet.make_image(width, height, 3)                                               #Creating empty image file for yolo detection input
    darknet.copy_image_from_bytes(img_for_detect, frame_rgb_resized.tobytes())                          #Copying resized rgb image to 'img_for_detect' instance for YOLO
        
    #run detections
    detections = darknet.detect_image(network, class_names, img_for_detect, thresh=0.5)                 #Run detect_image module from darknet.py and out put detections with array of 6 elements;
                                                                                                        #detections = [ label , confidence , bbox co-ords containing 4 elements x,y,w,h ]
        
       
    #Counting number of each class detected
    for obj in detections :
        if obj[0] == 'windows_installed' : 
            windows_installed_count += 1
            print("\nInstalled Window Detected!")
            
        elif obj[0] == 'windows_uninstalled' : 
            windows_uninstalled_count += 1
            print("\nUninstalled Window Detected!")
            
        elif obj[0] == 'doors_uninstalled' : 
            doors_uninstalled_count += 1
            print("\nUninstalled Door Detected!")
            
        elif obj[0] == 'doors_installed' : 
            doors_installed_count += 1
            print("\nInstalled Door Detected!")
            
        elif obj[0] == 'electrical_power' : 
            electrical_power_count += 1
            print("\nElectrical power outlet Detected!")
            
        elif obj[0] == 'electrical_switch' : 
            electrical_switch_count += 1
            print("\nElectrical Switch Detected!")
            
        elif obj[0] == 'electrical_telecom' : 
            electrical_telecom_count += 1
            print("\nElectrical Telecom outlet Detected!")
            
        elif obj[0] == 'electrical_lights' : 
            electrical_lights_count += 1
            print("\nElectrical Lighting Detected!")
            
        elif obj[0] == 'electrical_uninstalled' : 
            electrical_uninstalled_count += 1
            print("\nUninstalled electrical points Detected!")
            
        elif obj[0] == 'electrical_mains' : 
            electrical_mains_count += 1
            print("\nElectrical Circuit box Detected!")
            
        elif obj[0] == 'pvc_pipes' : 
            pvc_pipes_count += 1
            print("\nPVC Pipe(s) Detected!")
            
        elif obj[0] == 'cement_bag' : 
            cement_bag_count += 1
            print("\nCement Bag Detected!")
            
        elif obj[0] == 'exit_signage' : 
            exit_signage_count += 1
            print("\nExit Sign Detected!")
            
        elif obj[0] == 'wires' : 
            wires_count += 1
            print("\nWire(s) Detected!")                
        
        
        
        
    #draw bounding boxes on detections
    image = darknet.draw_boxes(detections, frame_rgb_resized, class_colors)                             #image = detections array, frame_rgb_resized=detected image at current frame, colors of labels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #Displaying livestream detection in a window (graphical representation)
    #cv2.imshow('Inventory Detections', image)                                 #cv2.imshow("display window name", "video frame feed")
    cv2.waitKey(1)                                                           #Displaying video for time in milliseconds per frame (controlling FPS) 


#Stop streaming video
cap.release()
#out.release()
cv2.destroyAllWindows()

#---------------------Printing Inventory List of Detected Objects-----------------------#
print("\n\nDetections Completed!")
print("\nInventory List")
line1 = str(print("\nTotal No. of Windows Installed: ", windows_installed_count))
line2 = str(print("\nTotal No. of Windows Uninstalled: ", windows_uninstalled_count))
line3 = str(print("\nTotal No. of Doors Installed: ", doors_installed_count))
line4 = str(print("\nTotal No. of Doors Uninstalled: ", doors_uninstalled_count))
line5 = str(print("\nTotal No. of Electrical Power Points: ", electrical_power_count))
line6 = str(print("\nTotal No. of Electrical Switches: ", electrical_switch_count))
line7 = str(print("\nTotal No. of Electrical Telecom Points: ", electrical_telecom_count))
line8 = str(print("\nTotal No. of Electrical Lights: ", electrical_lights_count))
line9 = str(print("\nTotal No. of Electrical Wires/Cables: ", wires_count))
line10 = str(print("\nTotal No. of Uninstalled Electrical Points: ", electrical_uninstalled_count))
line11 = str(print("\nTotal No. of Electrical Circuit Boxes: ", electrical_mains_count))
line12 = str(print("\nTotal No. of PVC Pipes: ", pvc_pipes_count))
line13 = str(print("\nTotal No. of Cement bags: ", cement_bag_count))
line14 = str(print("\nTotal No. of Exit Signs: ", exit_signage_count))
#--------------------- Saving Inventory List into Text File-----------------------------#
print("Inventory List Saved as InvtList.txt")
txtfile = open("InvtList.txt", "w")
txtfile.write("\nInventory List\n\n")
txtfile.writelines([line1,line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12,line13,line14])
txtfile.write('\n'+ 'Total No. of Windows Installed: ' + windows_installed_count)
# txtfile.write("\nTotal No. of Windows Uninstalled: ", windows_uninstalled_count)
# txtfile.write("\nTotal No. of Doors Installed: ", doors_installed_count)
# txtfile.write("\nTotal No. of Doors Uninstalled: ", doors_uninstalled_count)
# txtfile.write("\nTotal No. of Electrical Power Points: ", electrical_power_count)
# txtfile.write("\nTotal No. of Electrical Switches: ", electrical_switch_count)
# txtfile.write("\nTotal No. of Electrical Telecom Points: ", electrical_telecom_count)
# txtfile.write("\nTotal No. of Electrical Lights: ", electrical_lights_count)
# txtfile.write("\nTotal No. of Electrical Wires/Cables: ", wires_count)
# txtfile.write("\nTotal No. of Uninstalled Electrical Points: ", electrical_uninstalled_count)
# txtfile.write("\nTotal No. of Electrical Circuit Boxes: ", electrical_mains_count)
# txtfile.write("\nTotal No. of PVC Pipes: ", pvc_pipes_count)
# txtfile.write("\nTotal No. of Cement bags: ", cement_bag_count)
# txtfile.write("\nTotal No. of Exit Signs: ", exit_signage_count)
txtfile.close()
