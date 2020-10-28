unsigned long aktualnyCzas = 0;
unsigned long zapamietanyCzas = 0;
unsigned long roznicaCzasu = 0;

const int red0 =  2;// the number of the LED pin
const int yellow0 =  3;// the number of the LED pin
const int green0 =  4;// the number of the LED pin
const int red1 =  5;// the number of the LED pin
const int yellow1 =  6;// the number of the LED pin
const int green1 =  7;// the number of the LED pin
const int red2 =  8;// the number of the LED pin
const int yellow2 =  9;// the number of the LED pin
const int green2 =  10;// the number of the LED pin
const int button = 11;
int buttonState = 0;  
int counter = 0;
int lastState = HIGH; // the previous state from the input pin
int currentState;    // the current reading from the input pin

void setup(){
  pinMode(red0, OUTPUT);
  pinMode(yellow0, OUTPUT);
  pinMode(green0, OUTPUT);
  pinMode(red1, OUTPUT);
  pinMode(yellow1, OUTPUT);
  pinMode(green1, OUTPUT);
  pinMode(red2, OUTPUT);
  pinMode(yellow2, OUTPUT);
  pinMode(green2, OUTPUT);
  pinMode(button, INPUT_PULLUP);
  Serial.begin(9600);
}

void loop(){
  currentState = digitalRead(button);
//  Serial.println(buttonState);
   if (currentState == HIGH && lastState == LOW && counter < 4) {
    counter ++;
    Serial.println(counter);
   }
   else if(currentState == HIGH && lastState == LOW && counter==4){
    counter = 0;
    Serial.println(counter);
   }
   switch (counter){
    case 0:
      traffic_sync();
      break;

    case 1:
      green();
      break;

    case 2:
      yellow();
      break;
    
    case 3:
      red_yellow();
      break;
    
    case 4:
      red();
      break;
  }
   lastState = currentState;
}

void traffic_sync(){
    lastState = currentState;
  //Pobierz liczbe milisekund od startu
  aktualnyCzas = millis();
  roznicaCzasu = aktualnyCzas - zapamietanyCzas;
  if (roznicaCzasu >= 0 && roznicaCzasu <5000UL){
    digitalWrite(green0, HIGH);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, LOW);
    
    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);
    
    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  
  //Jeśli różnica wynosi ponad sekundę
  if (roznicaCzasu >= 5000UL && roznicaCzasu <7000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, HIGH);
    digitalWrite(red0, LOW);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu >= 7000UL && roznicaCzasu <9000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, HIGH);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu >= 9000UL && roznicaCzasu <14000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, HIGH);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, LOW);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu >= 14000UL && roznicaCzasu <16000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, HIGH);
    digitalWrite(red1, LOW);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu >= 16000UL && roznicaCzasu <18000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, HIGH);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu >= 18000UL && roznicaCzasu <23000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, HIGH);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, LOW);
  }
  if (roznicaCzasu >= 23000UL && roznicaCzasu <25000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, HIGH);
    digitalWrite(red2, LOW);
  }
  if (roznicaCzasu >= 25000UL && roznicaCzasu <27000UL) {
//    zapamietanyCzas = aktualnyCzas;
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, HIGH);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
  }
  if (roznicaCzasu > 27000UL){
    zapamietanyCzas = aktualnyCzas;
  }
}

void green(){
    digitalWrite(green0, HIGH);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, LOW);

    digitalWrite(green1, HIGH);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, LOW);

    digitalWrite(green2, HIGH);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, LOW);
}


void yellow(){
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, HIGH);
    digitalWrite(red0, LOW);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, HIGH);
    digitalWrite(red1, LOW);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, HIGH);
    digitalWrite(red2, LOW);
}

void red_yellow(){
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, HIGH);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, HIGH);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, HIGH);
    digitalWrite(red2, HIGH);
}


void red(){
    digitalWrite(green0, LOW);
    digitalWrite(yellow0, LOW);
    digitalWrite(red0, HIGH);

    digitalWrite(green1, LOW);
    digitalWrite(yellow1, LOW);
    digitalWrite(red1, HIGH);

    digitalWrite(green2, LOW);
    digitalWrite(yellow2, LOW);
    digitalWrite(red2, HIGH);
}
