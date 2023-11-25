import serial

ser = serial.Serial("/dev/rfcomm0", 9600, timeout = 1) #Change your port name COM... and your baudrate

def sendData(data):
    ser.write(str(data).encode())
    

if __name__ == '__main__':
   while(True):
    uInput = input("Enviar data? ")
    ser.write(str(uInput).encode())
