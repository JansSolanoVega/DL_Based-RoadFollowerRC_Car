#include <Servo.h>

#include <SoftwareSerial.h>
SoftwareSerial SerialBT(11,12); 
//Declaramos el servo
Servo servo;

//Declaramos la variable
char dato;
#define enA 5
#define in1A 3
#define in2A 2
#define enB 6
#define in1B 7
#define in2B 8
#define SERVO 10

float vel_max=500;
int angulo = 30;

void setup() {
  Serial.begin(9600);
  SerialBT.begin(9600); //Bluetooth device name
  pinMode(enA, OUTPUT);
  pinMode(in1A, OUTPUT);
  pinMode(in2A, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1B, OUTPUT);
  pinMode(in2B, OUTPUT);
  // Set initial rotation direction
  digitalWrite(in1A, LOW);
  digitalWrite(in2A, LOW);
  digitalWrite(in1B, LOW);
  digitalWrite(in2B, LOW);
  
  
  servo.attach(SERVO);
  servo.write(angulo);
}
String text;
float primerValor;
float segundoValor;
int dir;
void loop() {
  if (SerialBT.available()) {
    text = SerialBT.readStringUntil('\n');
    //Serial.println(text);
    
    int i1 = text.indexOf(',');
  
    String firstValue = text.substring(0, i1);
    String secondValue = text.substring(i1 + 1);
    primerValor=firstValue.toFloat();
    segundoValor=secondValue.toFloat();
    
    Serial.print(int(primerValor));
    Serial.print(",");
    Serial.println(int(vel_max*segundoValor));
        
    if(segundoValor>0){
      dir=1;
    }
    else{
      dir=-1;
    }
    motorTrasero(abs(vel_max*segundoValor),dir);
    motorDelantero(abs(vel_max*segundoValor),dir);
    
    servo.write(int(primerValor));
  }

  
}

void motorDelantero(int vel,int avanza_stop_retro){

  int pwmOutput = map(vel, 0, 1023, 0 , 255); // Map the potentiometer value from 0 to 255
  analogWrite(enA, pwmOutput); // Send PWM signal to L298N Enable pin

  if (avanza_stop_retro==1) {
    digitalWrite(in1A, HIGH);
    digitalWrite(in2A, LOW);
  }
  if (avanza_stop_retro==0) {
    digitalWrite(in1A, LOW);
    digitalWrite(in2A, LOW);
  }
  if (avanza_stop_retro==-1) {
    digitalWrite(in1A, LOW);
    digitalWrite(in2A, HIGH);
  }
}

void motorTrasero(int vel,int avanza_stop_retro){

  int pwmOutput = map(vel, 0, 1023, 0 , 255); // Map the potentiometer value from 0 to 255
  analogWrite(enB, pwmOutput);
  if (avanza_stop_retro==1) {
    digitalWrite(in1B, HIGH);
    digitalWrite(in2B, LOW);
  }
  if (avanza_stop_retro==0) {
    digitalWrite(in1B, LOW);
    digitalWrite(in2B, LOW);
  }
  if (avanza_stop_retro==-1) {
    digitalWrite(in1B, LOW);
    digitalWrite(in2B, HIGH);
  }
}
