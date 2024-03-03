/*
 * HC-SR04 example sketch
 *
 * https://create.arduino.cc/projecthub/Isaac100/getting-started-with-the-hc-sr04-ultrasonic-sensor-036380
 *
 * by Isaac100
 */

// *Interfacing RGB LED with Arduino 
// * Author: Osama Ahmed 

//Defining  variable and the GPIO pin on Arduino
int redPin= 11;
int greenPin = 9;
int  bluePin = 10;

int value = 0;

const int trigPin1 = 7;
const int echoPin1 = 8;

const int trigPin2 = 5;
const int echoPin2 = 4;

float duration1, duration2, distance1, distance2;

void setup() {
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);
  pinMode(redPin,  OUTPUT);              
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
  Serial.begin(9600);
  setColor(0, 0, 0); 
}

void loop() {
  value = Serial.read();
  if (value == 1)
  {
    setColor(80, 210, 255); 
  }
  if (value == 0)
  {
    setColor(0, 0, 0); 
  }

  Serial.println(value);

  digitalWrite(trigPin1, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin1, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin1, LOW);

  duration1 = pulseIn(echoPin1, HIGH);
  distance1 = (duration1*.0343)/2;
  Serial.print("Distance1 ");
  Serial.println(distance1);
  delay(30);

  digitalWrite(trigPin2, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin2, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin2, LOW);
  

  duration2 = pulseIn(echoPin2, HIGH);
  distance2 = (duration2*.0343)/2;
  Serial.print("Distance2 ");
  Serial.println(distance2);
  delay(30);
}

void setColor(int redValue, int greenValue,  int blueValue) {
  analogWrite(redPin, 255 - redValue);
  analogWrite(greenPin,  255 - greenValue);
  analogWrite(bluePin, 255 - blueValue);
}

