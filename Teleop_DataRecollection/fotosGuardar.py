import cv2
from serial import Serial
import pandas as pd

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


def show_camera():
    window_title = "CSI Camera"

    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            cont=0
            lista=[]
            while True:
                ret_val, frame = video_capture.read()
                if arduino.in_waiting:
                    data = arduino.readline().decode('utf-8').rstrip()
                    angulo=data.split(",")[0]
                    velocidad=data.split(",")[1]
                    nombre_imagen="/home/mecatronicag3/Desktop/temasCArduino/fotos/"+'Img'+str(cont)+'.png'
                    lista.append({
                        'Nombre_imagen': nombre_imagen,
                        'Angulo': angulo,
                        'Velocidad': velocidad
                    })
                    print(lista)
                    df = pd.DataFrame(lista)
                    df.to_csv("/home/mecatronicag3/Desktop/temasCArduino/"+"dataFrame.csv")
                    cv2.imwrite(nombre_imagen,frame)
                    cont+=1     
                    
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    arduino = Serial('/dev/ttyACM0', 9600, timeout=1)
    show_camera()
