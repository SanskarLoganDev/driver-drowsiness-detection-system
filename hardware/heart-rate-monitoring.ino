// This code is supposed to be run Arduino IDE and will show error if added in any other IDE like VS Code as the other
// modules/hardware is not connected

#include <ESP8266WiFi.h>;
#include <WiFiClient.h>; 
#include <ThingSpeak.h>; 

const char* ssid = "OPPO F7 128GB"; //Your Network SSID 
const char* password = "12345678"; //Your Network Password 
int val; 
int PulseSensorpin = A0; //Pulse Sensor Pin Connected at A0 Pin 
WiFiClient client; 
unsigned long myChannelNumber = 1330026 ; //Your Channel Number (Without Brackets) 

const char * myWriteAPIKey = "49JFMCWGZ6V1MKCZ"; //Your Write API Key 
void setup() 
{ 
    Serial.begin(9600); 
    delay(10); 
    // Connect to WiFi network 
    WiFi.begin(ssid, password); 
    ThingSpeak.begin(client); 
} 
void loop() 
{ 
    val = analogRead(PulseSensorpin); //Read Analog values and Store in val variable 
    Serial.println("Pulse Sensorvalue= "); // Start Printing on Pulse sensor value on LCD 
    Serial.println(val*100/1023); // Start Printing on Pulse sensor value on LCD 
    delay(400); 
    ThingSpeak.writeField(myChannelNumber, 1,val*100/1023, myWriteAPIKey); //Update in 
    ThingSpeak 
}