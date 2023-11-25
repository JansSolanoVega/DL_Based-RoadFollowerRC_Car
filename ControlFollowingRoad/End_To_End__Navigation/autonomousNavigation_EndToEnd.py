import cv2
import sys
from serial import Serial
import pandas as pd
sys.path.append("/home/mecatronicag3/venvs/temasc/lib/python3.6/site-packages")
sys.path.append("/home/mecatronicag3/torchvision")
import torch
from torch import nn
from torch.nn import Flatten, Conv2d, Linear, ELU, Dropout, LogSoftmax

import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
import SerialModule
val=0
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3264,
    capture_height=2464,
    display_width=600,
    display_height=400,
    framerate=21,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class nvidiaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.elu = ELU()
        self.conv0 = Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2,2))
        self.conv1 = Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2,2))
        self.conv2 = Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2,2)) 
        self.conv3 = Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3)) 
        self.conv4 = Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)) 
        self.dropout = Dropout(p=0.5)
        self.fc1 = Linear(in_features=1152, out_features=100)
        self.fc2 = Linear(in_features=100, out_features=50)
        self.fc3 = Linear(in_features=50, out_features=10)
        self.fc4 = Linear(in_features=10, out_features=5)

    def forward(self, x):
        x = self.elu(self.conv0(x))
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.elu(self.conv3(x))
        x = self.elu(self.conv4(x))
        x = self.dropout(x)
        x = self.flatten(x)
        #FCN
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.elu(self.fc2(x))
        x = self.dropout(x)
        x = self.elu(self.fc3(x))
        x = self.dropout(x)
        logits = self.fc4(x)
        return logits

class Preprocessing:
    def __init__(self, porcentaje_cropped):
        self.porcentaje = porcentaje_cropped

    def __call__(self, x):
        img_array = np.asarray(x)
        W, H, C = img_array.shape
        img = img_array[int(self.porcentaje*W):,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img / 255
        return img
    
class ToFloat32:
    def __call__(self, x):
        return x.to(torch.float32)
transform = transforms.Compose([
    Preprocessing(porcentaje_cropped = 0.2),
    transforms.ToTensor(),
    ToFloat32(),
])

modelo = nvidiaModel()
modelo.load_state_dict(torch.load("modelTerrenatorNvidia.pth"))
modelo.eval()

flag = ""

while True:

    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            cont=0
            lista=[]
            b=time.time()
            while True:
                
                ret_val, frame = video_capture.read()
                
                #Evaluando el modelo
                a=time.time()
                color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image=Image.fromarray(color_converted)
                img = transform(pil_image)

                with torch.no_grad():
                    output = modelo(img.unsqueeze(0))
                    _, predicted = torch.max(output.data, 1)
                    print("Clase predicha: ", str(predicted.item()))
                    print("Tiempo de ejecucion: ", str(time.time()-a))
                    print("FPS: ", str(int(1/(time.time()-a)))) 
                    velocidad=0.60
                    ang_servo=30*(int(predicted)+1)
                    if ang_servo ==30 or ang_servo==90:
                        velocidad=0.85
                    
                toSend = str(ang_servo)+","+str(velocidad)+"\n"
                
                if time.time()-b>0.35:
                    b=time.time()
                    SerialModule.sendData(toSend)       

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
            velocidad=0
            ang_servo=60
            
            toSend = str(ang_servo)+","+str(velocidad)+"\n"
            SerialModule.sendData(toSend) 
            break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

SerialModule.cerrar()

