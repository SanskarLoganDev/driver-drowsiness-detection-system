# driver-drowsiness-detection-system
This project demonstrates three different non-invasive methods of drowsiness detection namely  Eye-drowsiness Detection 
using Opencv and Aspect Ratio , Yawning detection using  Euclidean distance between both lips , Pulse Rate Measurement 
and its Central Monitoring  System.

1. Driver Alarming System: This System would continuously capture the image of the eyes of the 
drivers using Open Cv and Machine learning models. These images will be matched to the trained 
dataset. 
Then the Alarming System will generate a drowsiness score of the drivers. If the drowsiness level 
exceeds the threshold value, then the driver will be detected as drowsy and alarm will ring which 
will alert driver.

2. Yawning Detection System: This System will detect the yawn of the drivers as yawning indicates 
the drivers might fall asleep while driving vehicles. It will generate the yawning score from range 
of 1 to 10. If the yawning score is maximum i.e. 10 then the driver is detected is yawning.

3. Pulse Rate Detection: Pulse rate of the driver is continuously monitored using Pulse Sensor tied to 
the driver’s wrist. The data will be send to the cloud using Arduino IDE through Node MCU.

4. Central Monitoring System: The project consists of Central Cloud servers which monitors the heart 
rate and plot its corresponding graph using the help of thingspeak. The change in slope of the pulse 
rate will indicate the sleeping tendency of the drivers as pulse rate is inversely proportional to the 
drowsiness. This system will ensure the safety of drivers.
