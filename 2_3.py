from __future__ import print_function
import cv2
import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import multiprocessing
import sys
def restrict(color_component):
    return np.clip(color_component, 0, 255)

def calculate(frame_1):
    low = []
    high = []
    top_crop_y = 20
    bottom_crop_y = 359
    f_flattened = []
    f_hsv = cv2.cvtColor(frame_1, cv2.COLOR_RGB2HSV)

    for index in range(3):
        top = np.array(f_hsv[:top_crop_y, :, index]).flatten()
        bottom = np.array(f_hsv[bottom_crop_y:, :, index]).flatten()
        top_and_bottom = np.append(top, bottom)
        top_and_bottom_series = pd.Series(top_and_bottom)
        #print(top_and_bottom_series.describe())
        f_flattened.append(top_and_bottom_series)
    

    z_value = 10.0
    for i in range(3):
        mu = f_flattened[i].values.mean()
        sigma = f_flattened[i].values.std()
        deviation = z_value*sigma
        #print(mu)
        #print(deviation)
        low.append(restrict(mu-deviation-20))
        high.append(restrict(mu+deviation+20))

    return low, high


def chroma_key(frame_1, frame_2, low, high):
    try:
        bg_cropped = frame_2[:len(frame_1), :len(frame_1[0]), :]
    except:
        bg_cropped=0
        
 #   try:
    #plt.imshow(frame_1)
    #print(frame_1.shape)
    #plt.show()
    #print(low)
    #print(high)
    mask_lower = np.array([low[0], low[1], low[2]])
    mask_higher = np.array([ high[0], high[1], high[2]])
    f_hsv = cv2.cvtColor(frame_1, cv2.COLOR_RGB2HSV)
    f_mask = cv2.inRange(f_hsv, mask_lower, mask_higher)

    masked_f = np.copy(frame_1)
    masked_f[f_mask != 0] = [0, 0, 0]

    f_hand = np.copy(frame_1)
    masked_f[f_mask != 0] = [0, 0, 0]

    bg_masked = np.copy(bg_cropped)
    bg_masked[f_mask == 0] = [0,0,0]

    full_picture = bg_masked + masked_f
    #cv2.imshow('f_1',full_picture)
    return full_picture
#  except:
  #      return None
'''
if args.input_1 == None or args.input_2 == None:
    print("Please give input file")
    exit(0)
else:
    v1 = cv2.VideoCapture(video1)
    v2 = cv2.VideoCapture(video2)
    width = v1.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = v1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = v1.get(cv2.CAP_PROP_FPS)
    ov = cv2.VideoWriter(args.output, 0,fps, (int(width),int(height)))
    i=0
    #calculate(cv2.imread(args.background))
    while v1.isOpened() and v2.isOpened():
        ret_1, frame_1 = v1.read()
        ret_2, frame_2 = v2.read()
        if i==0:
            calculate(frame_1)
            i+=1
        full_picture = chroma_key(frame_1, frame_2)
        ov.write(full_picture)
        #cv2.imshow('frame_1', frame_1)
        #cv2.imshow('frame_2', frame_2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    v1.release()
    cv2.destroyAllWindows()
'''
def get_video(iteration, fps, width, height, low, high):
    print("Now doing: ", iteration)
    #sys.stdout.flush()
    v1 = cv2.VideoCapture(str(iteration) + "v1.mp4")
    v2 = cv2.VideoCapture(str(iteration) + "v2.mp4")
    ov = cv2.VideoWriter(str(iteration) + "output.mp4", 0,fps, (int(width),int(height)))
    i = 0
    while v1.isOpened() and v2.isOpened():
        ret_1, frame_1 = v1.read()
        ret_2, frame_2 = v2.read()
        #if i==0 and iteration == 1:
        #    low, high = calculate(frame_1)
        #   i+=1
        if not isinstance(frame_1, np.ndarray):
            break
        full_picture = chroma_key(frame_1, frame_2, low, high)
        ov.write(full_picture)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ov.release()
    v1.release()
    v2.release()
    cv2.destroyAllWindows()
    #return low, high
if __name__ == "__main__":

    video1 = r"foot.mp4"
    video2 = r"back.mp4"

    v1 = cv2.VideoCapture(video1)
    v2 = cv2.VideoCapture(video2)
    global width
    width = v1.get(cv2.CAP_PROP_FRAME_WIDTH)
    global height
    height = v1.get(cv2.CAP_PROP_FRAME_HEIGHT)
    global fps
    fps = v1.get(cv2.CAP_PROP_FPS)
    length1 = int(v1.get(cv2.CAP_PROP_FRAME_COUNT))
    length2 = int(v1.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length1, length2)
    
    #Dividing videos
    count = 0
    itercount = 1
    jump = length1 // 8
    while itercount <= 8:
        print("Iteration: ", itercount)
        
        iteration1 = str(itercount) + "v1.mp4"
        iteration2 = str(itercount) + "v2.mp4"
        ov1 = cv2.VideoWriter(iteration1, 0,fps, (int(width),int(height)))
        ov2 = cv2.VideoWriter(iteration2, 0,fps, (int(width),int(height)))
        i = 0
        while i < jump:
            ret_1, frame_1 = v1.read()
            ret_2, frame_2 = v2.read()
            if i == 0 and itercount == 1:
                low, high = calculate(frame_1)
            if not isinstance(frame_1, np.ndarray):
                break

            ov1.write(frame_1)
            ov2.write(frame_2)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
        if itercount == 8:
            i = 0
            while i < length1 % 8:
                ret_1, frame_1 = v1.read()
                ret_2, frame_2 = v2.read()
                
                if not isinstance(frame_1, np.ndarray):
                    break

                ov1.write(frame_1)
                ov2.write(frame_2)

                i += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                count += 1
        itercount += 1
        ov1.release()
        ov2.release()
    v1.release()
    v2.release()
    cv2.destroyAllWindows()

    #Processing sub videos
    processes = []
    for i in range(1, 9):
        p = multiprocessing.Process(target = get_video, args=(i, fps, width, height, low, high, ))
        processes.append(p)
        p.start()
        
    for process in processes:
        process.join()

    #Appending videos
    cap = cv2.VideoCapture("1output.mp4")
    video_index = 0
    out = cv2.VideoWriter("output.avi", 0, fps, (int(width),int(height)))
    videofiles = [str(i) + "output.mp4" for i in range(1, 9)]
    while(cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            print("end of video " + str(video_index) + " .. next one now")
            video_index += 1
            if video_index >= len(videofiles):
                break
            cap = cv2.VideoCapture(videofiles[ video_index ])
            ret, frame = cap.read()
        cv2.imshow('frame',frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
