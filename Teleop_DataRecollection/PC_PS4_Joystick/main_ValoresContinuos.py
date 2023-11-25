import BlueToothModule as bM
import JoyStickModule as jsM
import cv2
from time import sleep

while(True):
    velocidad= jsM.getJS('axis4')*-1*0.4
    x_controller = jsM.getJS('axis1')
    ang_servo = 60- x_controller*30
    print('ang_servo: ' + str(ang_servo) + ', ' + 'velocidad: ' + str(velocidad)+'\n')
    toSend = str(ang_servo) + ','+ str(velocidad)+'\n'
    bM.sendData(toSend)
    sleep(0.2)


    

