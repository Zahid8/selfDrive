#define L_PWM 5
#define L_DIR 4
#define R_PWM 6
#define R_DIR 9
#define PWM_MAX 165

#define BUZZER 10
#define LEFT_BLINKER 12
#define RIGHT_BLINKER 8
#define STOP_LIGHT 7
#define TSOP_PIN 3


const char startOfNumberDelimiter = '<';
const char endOfNumberDelimiter   = '>';


int leftSpeed = 0;
int rightSpeed = 0;


void setup() {
  //Konfiguracja pinow od mostka H
  pinMode(L_DIR, OUTPUT);
  pinMode(R_DIR, OUTPUT);
  pinMode(L_PWM, OUTPUT);
  pinMode(R_PWM, OUTPUT);

  //Konfiguracja pozostalych elementow
  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, 0); //Wylaczenie buzzera  
  pinMode(LEFT_BLINKER, OUTPUT); 
  digitalWrite(LEFT_BLINKER, 0); //Wylaczenie diody
  pinMode(RIGHT_BLINKER, OUTPUT); 
  digitalWrite(RIGHT_BLINKER, 0); //Wylaczenie diody
  pinMode(STOP_LIGHT, OUTPUT); 
  digitalWrite(STOP_LIGHT, 0); //Wylaczenie diody
  
  Serial.begin(9600);
  
}

void loop() {
//  digitalWrite(LED, HIGH);
//  delay(1000);                       // wait for a second
//  digitalWrite(LED, LOW);    // turn the LED off by making the voltage LOW
//  delay(1000);

 while (Serial.available()){
  

    processInput();
//    Serial.println(leftSpeed);
//    Serial.println(rightSpeed);

    leftMotor(leftSpeed);
    rightMotor(rightSpeed);
}
}
void lights(String input){
  char lightType = input.charAt(1);
  char lightState = input.charAt(2);
  if (lightType == '0' && lightState == '0'){
    digitalWrite(STOP_LIGHT, LOW);
  }
  else if(lightType == '0' && lightState == '1'){
    digitalWrite(STOP_LIGHT, HIGH);
  }
                                                                                                                                                                                                               
}
void setMotorSpeed(String input){
  char leftSign = input.charAt(1);
  String leftString = input.substring(2,5);
  if (leftSign == '0'){
    leftSpeed = -leftString.toInt();
  }
  else{
    leftSpeed = leftString.toInt();
  }
  char rightSign = input.charAt(5);
  String rightString = input.substring(6,9);
  if (rightSign == '0'){
    rightSpeed = -rightString.toInt();
  }
  else{
   rightSpeed = rightString.toInt();
  }
}


void processNumber (const long n)  {
  String input = String(n);
  char module = input.charAt(0);
  if (module == '1'){
    setMotorSpeed(input);
  }
  else if (module == '2'){
    lights(input);
  }

}  // end of processNumber
  
void processInput ()
  {
  static long receivedNumber = 0;
  static boolean negative = false;
  
  byte c = Serial.read ();
  
  switch (c)
    {
      
    case endOfNumberDelimiter:  
      if (negative) 
        processNumber (- receivedNumber); 
      else
        processNumber (receivedNumber); 

    // fall through to start a new number
    case startOfNumberDelimiter: 
      receivedNumber = 0; 
      negative = false;
      break;
      
    case '0' ... '9': 
      receivedNumber *= 10;
      receivedNumber += c - '0';
      break;
      
    case '-':
      negative = true;
      break;
      
    } // end of switch  
  }  // end of processInput

void leftMotor(int V) {
  if (V > 100){
    V = 100;
  }
  if (V > 0) { //Jesli predkosc jest wieksza od 0 (dodatnia)
    V = map(V, 0, 100, 0, PWM_MAX);
    digitalWrite(L_DIR, 0); //Kierunek: do przodu
    analogWrite(L_PWM,V); //Ustawienie predkosci 
  } else {
    V = abs(V); //Funkcja abs() zwroci wartosc V  bez znaku
    V = map(V, 0, 100, 0, PWM_MAX);
    digitalWrite(L_DIR, 1); //Kierunek: do tyłu
    analogWrite(L_PWM,V); //Ustawienie predkosci
        
  }
//  return S;
}

void rightMotor(int V) {
  if (V > 100){
    V = 100;
  }
  if (V > 0) { //Jesli predkosc jest wieksza od 0 (dodatnia)
    V = map(V, 0, 100, 0, PWM_MAX);
    digitalWrite(R_DIR, 0); //Kierunek: do przodu
    analogWrite(R_PWM, V); //Ustawienie predkosci 
  } else {
    V = abs(V); //Funkcja abs() zwroci wartosc V  bez znaku
    V = map(V, 0, 100, 0, PWM_MAX);
    digitalWrite(R_DIR, 1); //Kierunek: do tyłu
    analogWrite(R_PWM, V); //Ustawienie predkosci 
  }
//  return S;
}

void stopMotors() {
  analogWrite(L_PWM, 0); //Wylaczenie silnika lewego
  analogWrite(R_PWM, 0); //Wylaczenie silnika prawego
}
