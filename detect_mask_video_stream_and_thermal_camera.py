#initialize buzzer
from gpiozero import Buzzer
buzzer = Buzzer(23)

# import the necessary packages
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os

import time,sys
sys.path.append('../')
# load AMG8833 module
import amg8833_i2c
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import interpolate

import warnings
warnings.simplefilter("ignore", DeprecationWarning)


#
#####################################
# Initialization of Sensor
#####################################
#
# init calib parameter for thermal sensor
calib = 7.4

t0 = time.time()
sensor = []
while (time.time()-t0)<1: # wait 1sec for sensor to start
    try:
        # AD0 = GND, addr = 0x68 | AD0 = 5V, addr = 0x69
        sensor = amg8833_i2c.AMG8833(addr=0x69) # start AMG8833
    except:
        sensor = amg8833_i2c.AMG8833(addr=0x68)
    finally:
        pass
time.sleep(0.1) # wait for sensor to settle

# If no device is found, exit the script
if sensor==[]:
    print("No AMG8833 Found - Check Your Wiring")
    sys.exit(); # exit the app if AMG88xx is not found 

#
#####################################
# Interpolation Properties 
#####################################
#

# original resolution
pix_res = (8,8) # pixel resolution
xx,yy = (np.linspace(0,pix_res[0],pix_res[0]),
         np.linspace(0,pix_res[1],pix_res[1]))
zz = np.zeros(pix_res) # set array with zeros first

# new resolution
pix_mult = 6 # multiplier for interpolation 
interp_res = (int(pix_mult*pix_res[0]),int(pix_mult*pix_res[1]))
grid_x,grid_y = (np.linspace(0,pix_res[0],interp_res[0]),
                 np.linspace(0,pix_res[1],interp_res[1]))

# interp function
def interp(z_var):
    # cubic interpolation on the image
    # at a resolution of (pix_mult*8 x pix_mult*8)
    f = interpolate.interp2d(xx,yy,z_var,kind='cubic')
    return f(grid_x,grid_y)

grid_z = interp(zz) # interpolated image

#
#####################################
# Start and Format Figure 
#####################################
#
plt.rcParams.update({'font.size':8})
fig_dims = (5,5) # figure size
fig,ax = plt.subplots(figsize=fig_dims) # start figure
#fig.canvas.set_window_title('AMG8833 Image Interpolation')
im1 = ax.imshow(grid_z,vmin=28,vmax=40,cmap=plt.cm.RdBu_r) # plot image, with temperature bounds
cbar = fig.colorbar(im1,fraction=0.046,pad=0.03) # colorbar
cbar.set_label('Temperature [*C]',labelpad=10) # temp. label
fig.canvas.draw() # draw figure

#ax_bgnd = fig.canvas.copy_from_bbox(ax.bbox) # background for speeding up runs
#fig.show() # show figure



# Open the camera
#cap = cv2.VideoCapture(0)
# Set initial value of weights
alpha = 0

#
#####################################
# Plot AMG8833 temps in real-time
#####################################
#
pix_to_read = 64 # read all 64 pixels



def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            #if face.any():
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)





# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
    default="face_detector",
    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
    default="mask_detector.model",
    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    status,pixels = sensor.read_temp(pix_to_read) # read pixels with status
    if status: # if error in pixel, re-enter loop and try again
        continue
    
    new_pixels = []
    for x in pixels:
        new_pixels.append(x + calib)
    
    #max_suhu = np.amax(pixels) + calib
    max_suhu = max(new_pixels)
    
    print('suhu:', max_suhu, '*C')
    
    #fig.canvas.restore_region(ax_bgnd) # restore background (speeds up run)
    new_z = interp(np.reshape(new_pixels,pix_res)) # interpolated image
    im1.set_data(new_z) # update plot with new interpolated temps
    ax.draw_artist(im1) # draw image again
    #fig.canvas.blit(ax.bbox) # blitting - for speeding up run
    #fig.canvas.flush_events() # for real-time plot
    
    # convert canvas to image
    #print(fig.canvas.tostring_rgb())
    canvas2img = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8,sep='')
    canvas2img = canvas2img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
     # img is rgb, convert to opencv's default bgr
    canvas2img = cv2.cvtColor(canvas2img,cv2.COLOR_RGB2BGR)
    
    # display image AMG8833 thermal
    cv2.imshow("AMG8833 Thermal Image w/ Interpolation",canvas2img)
    

    
    # Cropping an image as foreground
    cropped_img = canvas2img[72:430, 63:421] # img[y:y+h, x:x+w] 
    
    # Display cropped image
    #cv2.imshow("Cropped Image", cropped_img)
    
    

    # let's downscale the image using new  width and height
    resize_width = 150 
    resize_height = 150
    resize_points = (resize_width, resize_height)
    resized_img = cv2.resize(cropped_img, resize_points, interpolation=cv2.INTER_LINEAR)
    resized_img = cv2.flip(resized_img, 1)

    # Display images
    #cv2.imshow('Resized Down by defining height and width', resized_img)

    foreground = resized_img
    
    
    
    
    
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = cv2.flip(frame, 1)
    #frame = imutils.resize(frame, width=500)
    
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    buzzer.off() # kondisi awal buzzer
    
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        
        print('mask:', mask)
        print('withoutMask:', withoutMask)
    
        
        color_box = (0, 153, 0) # warna hijau
        # determine the class label and color we'll use to draw
        # the bounding box and text
        if mask > withoutMask:
            label1 = "Pakai Masker"
            color1 = (0, 153, 0) # warna hijau # referensi pilih warna https://www.rapidtables.com/web/color/RGB_Color.html
        else:
            label1 = "Tanpa Masker"
            color1 = (0, 0, 204) #warna merah
            color_box = (0, 0, 204) #warna merah
            buzzer.on()
        
        if max_suhu < 37.5:
            label2 = "Suhu Normal"
            color2 = (0, 153, 0) # warna hijau
        else:
            label2 = "Suhu Tinggi"
            color2 = (0, 0, 204) #warna merah
            color_box = (0, 0, 204) #warna merah
            buzzer.on()
            
        
        # include the probability in the label
        label1 = "{}: {:.2f}%".format(label1, max(mask, withoutMask)*100)
        label2 = "{}: {:.2f}*C".format(label2, max_suhu)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label1, (startX - 35, startY - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
        cv2.putText(frame, label2, (startX - 35, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color_box, 2)


    # create an overlay image.

    # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
    added_image = cv2.addWeighted(frame[0:resize_height, 0:resize_width], alpha, foreground, 1-alpha, 0)
     
    # Change the region with the result
    frame[0:resize_height, 0:resize_width] = added_image
    
    # show the output frame
    cv2.imshow("Program Pendeteksi Masker", frame)
    

        
        
        
        
    key = cv2.waitKey(1) & 0xFF

    # if the `e` key was pressed, break from the loop
    if key == ord("e"):
        buzzer.off()
        break

# do a bit of cleanup
buzzer.off()
cv2.destroyAllWindows()
vs.stop()