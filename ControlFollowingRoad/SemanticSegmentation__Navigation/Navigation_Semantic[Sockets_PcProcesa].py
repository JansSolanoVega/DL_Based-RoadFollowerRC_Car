import socket
import pickle
import struct
import cv2
import sys
from serial import Serial
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys
import BlueToothModule as bM
from torch.nn import Flatten, Conv2d, Linear, ELU, Dropout, LogSoftmax
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time

val=0
class Conv_3_k(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        return self.conv1(x)
class Double_Conv(nn.Module):
    '''
    Double convolution block for U-Net
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.double_conv = nn.Sequential(
                           Conv_3_k(channels_in, channels_out),
                           nn.BatchNorm2d(channels_out),
                           nn.ReLU(),
            
                           Conv_3_k(channels_out, channels_out),
                           nn.BatchNorm2d(channels_out),
                           nn.ReLU(),
                            )
    def forward(self, x):
        return self.double_conv(x)
    
class Down_Conv(nn.Module):
    '''
    Down convolution part
    '''
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.encoder = nn.Sequential(
                        nn.MaxPool2d(2,2),
                        Double_Conv(channels_in, channels_out)
                        )
    def forward(self, x):
        return self.encoder(x)
    
class Up_Conv(nn.Module):
    '''
    Up convolution part
    '''
    def __init__(self,channels_in, channels_out):
        super().__init__()
        self.upsample_layer = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bicubic'),
                        nn.Conv2d(channels_in, channels_in//2, kernel_size=1, stride=1)
                        )
        self.decoder = Double_Conv(channels_in, channels_out)
    
    def forward(self, x1, x2):
        '''
        x1 - upsampled volume
        x2 - volume from down sample to concatenate
        '''
        x1 = self.upsample_layer(x1)
        x = torch.cat([x2, x1],dim=1)
        return self.decoder(x)
    
class UNET(nn.Module):
    '''
    UNET model
    '''
    def __init__(self, channels_in, channels, num_classes):
        super().__init__()
        self.first_conv = Double_Conv(channels_in, channels) #64, 224, 224
        self.down_conv1 = Down_Conv(channels, 2*channels) # 128, 112, 112
        self.down_conv2 = Down_Conv(2*channels, 4*channels) # 256, 56, 56
        self.down_conv3 = Down_Conv(4*channels, 8*channels) # 512, 28, 28
        
        self.middle_conv = Down_Conv(8*channels, 16*channels) # 1024, 14, 14 
        
        self.up_conv1 = Up_Conv(16*channels, 8*channels)
        self.up_conv2 = Up_Conv(8*channels, 4*channels)
        self.up_conv3 = Up_Conv(4*channels, 2*channels)
        self.up_conv4 = Up_Conv(2*channels, channels)
        
        self.last_conv = nn.Conv2d(channels, num_classes, kernel_size=1, stride=1)
        
    def forward(self, x):
        x1 = self.first_conv(x)
        x2 = self.down_conv1(x1)
        x3 = self.down_conv2(x2)
        x4 = self.down_conv3(x3)
        
        x5 = self.middle_conv(x4)
        
        u1 = self.up_conv1(x5, x4)
        u2 = self.up_conv2(u1, x3)
        u3 = self.up_conv3(u2, x2)
        u4 = self.up_conv4(u3, x1)
        
        return self.last_conv(u4)

def getCentroid(image_np):
    x_c = 0
    y_c = 0
    
    area = image_np.sum()
    it = np.nditer(image_np, flags=['multi_index'])

    for i in it:
        x_c = i * it.multi_index[1] + x_c
        y_c = i * it.multi_index[0] + y_c

    (x_c,y_c) = int((x_c/area).astype(int)), int((y_c/area).astype(int))

    return x_c,y_c

transform_data = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()] )   

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

#MODELO
modeloCargado = UNET(3, 4, 2)
modeloCargado.load_state_dict(torch.load("AlvaroUnet.pth"))
modeloCargado.eval()

#SOCKETS
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.43.199'
port = 9993
client_socket.connect((host_ip, port))
data = b""
payload_size = struct.calcsize("Q")

cont=0
lista=[]
while True:
    c=time.time()
    while len(data) < payload_size:
        packet = client_socket.recv(4*1024)
        if not packet:
            break
        data += packet
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("Q", packed_msg_size)[0]

    while len(data) < msg_size:
        data += client_socket.recv(4*1024)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame = pickle.loads(frame_data)

    cv2.imshow("RECEIVING VIDEO", frame)

    #Evaluando el modelo
    a=time.time()
    color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image=Image.fromarray(color_converted)
    #img = transform(pil_image)
    tensor_frame = transform_data(pil_image)
    tensor_frame = tensor_frame.cuda()
    modeloCargado.cuda()
    
    with torch.no_grad():
        a=time.time()
        outputs = modeloCargado(tensor_frame.unsqueeze(0))
        preds = torch.argmax(outputs, dim=1).float()
        preds_cpu = preds.cpu()
        # Convertir el tensor a un arreglo NumPy de 2 dimensiones
        img_array = preds_cpu.numpy()[0]

        # Escalar los valores a un rango de 0 a 255
        img_array = (img_array - img_array.min()) * (255 / (img_array.max() - img_array.min()))

        # Convertir el arreglo a un formato de imagen válido para OpenCV
        img_array = img_array.astype('uint8')

        print("Tiempo de ejecucion: ", str(time.time()-a))
        print("FPS: ", str(int(1/(time.time()-a)))) 

        b=time.time()    
        (h, w)=img_array.shape

        x, y = getCentroid(img_array[int(w*0.4):,:])

        if x == np.nan:
            x = 0
        if y == np.nan:
            y = 0
        
        y+=int(w*0.4)
        '''
        x_m, y_m = int(h/2), w
        print("Centroide: ",x, y)
        diff = (x-x_m,-y+y_m)
        angle = cv2.fastAtan2(diff[1],diff[0])
        angle2 = np.arctan((x-x_m)/abs(y_m-y))*180.0/np.pi 
        if (angle == np.nan  or angle >180):
            angle=0

        if (angle2 == np.nan  or angle2 >180):
            angle2=0
    
        print("Anguloga: ", angle)
        print("Angulo2ga: ", angle2)

        EuclideanDistance = np.sqrt((x_m-x)*2+(y_m-y)*2)
        if (EuclideanDistance == np.nan or EuclideanDistance >500 ):
            EuclideanDistance=0
        '''
        bins = 3
        ancho = int(240/bins)
        val = int(x/ancho)
        velocidad=0.75

        val2 = bins-1-val
        '''
        ang_servo1=angle-30
        if ang_servo1<30:
            ang_servo1=30
        if ang_servo1>90:
            ang_servo1=90

        ang_servo2=angle2-30
        if ang_servo2<30:
            ang_servo2=30
        if ang_servo2>90:
            ang_servo2=90  
                                
        print("Angulo: ", ang_servo1)
        print("Angulo2: ", ang_servo2)
        '''
        a2=int(60/(bins-1))
        ang_servo1 = 30+val2*a2
        if ang_servo1==30 or ang_servo1==90:
            velocidad=1
        toSend = str(ang_servo1)+","+str(velocidad)+"\n"
        print(ang_servo1)
        bM.sendData(toSend)
        print(time.time()-c)
        im_cortada=img_array[int(w*0.4):,:]
        #cv2.circle(im_cortada, (x_m,y_m-int(w*0.4)), 5, (0,0,255), -1)
        #cv2.circle(im_cortada, (x,y-int(w*0.4)), 5, (0,0,255), -1)
        cv2.imshow('frame', im_cortada)
        if cv2.waitKey(1)==ord('q'):
            break   

    keyCode = cv2.waitKey(10) & 0xFF
        #Stop the program on the ESC key or 'q'
    if keyCode == 27 or keyCode == ord('q'):
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
client_socket.close()

'''
velocidad=0
ang_servo=60
toSend = str(ang_servo)+","+str(velocidad)+"\n"
bM.sendData(toSend) 

bM.cerrar()
'''
