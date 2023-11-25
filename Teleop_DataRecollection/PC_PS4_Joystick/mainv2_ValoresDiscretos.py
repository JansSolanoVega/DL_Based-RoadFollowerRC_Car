import BlueToothModule as bM
import JoyStickModule as jsM
import cv2
from time import sleep
import numpy as np
      
ang_servo,velocidad=60,0
while(True):
    val=0.9
    if jsM.getJS('t'):
        velocidad=val
    elif jsM.getJS('x'):
        velocidad=-1*val
    else:
        velocidad=0
    
    x_controller = jsM.getJS('axis1')
    ang = 60- x_controller*30

    num_bins=5
    _, bins = np.histogram([30,90], num_bins)
    
    for j in range(num_bins):
        if ang >= bins[j] and ang <= bins[j + 1]:
            position=j
            break
    
    scaled_position=position/(num_bins-1)
    ang_servo=30+scaled_position*60
    print('ang_servo: ' + str(ang_servo) + ', ' + 'velocidad: ' + str(velocidad)+'\n')
    toSend = str(ang_servo) + ','+ str(velocidad)+'\n'
    print(toSend)
    bM.sendData(toSend)
    sleep(0.2)


    

