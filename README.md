Sleep Detection:  

With Covid -19 and complete lockdown other then munching heavily on food , other important thing I am doing is sleeping bit more then I normally do , So as part of weekend project I decided to create Sleep Alert system (well I am using it for my son though 😊 ) 

It detects how long a given person’s eyes have been closed for. If their eyes have been closed for a certain amount of time, we’ll assume that they are starting to doze off and we will play an alarm to wake them up and grab their attention.

This is high level approach: 
1)	Using Dlib if face is found, we apply facial landmark detection 
2)	Extract eye region 
3)	Once we have extract Eye region, we can compute the eye aspect ratio 
4)	If eyes are closed for long time, alarm will sound 

Libraries Used: 
1)	Dlib : Sometime this is one of most weird library to install and after trying to resolve issues with Python 3.7 + I finally gave up and per me if you follow following steps that will be fastest and easiest… This is sometime turn out to be most crazy step 

Note : I used Conda Virtual Environment and install Python 3.6  along with other supporting libraries 
1.1	https://pypi.org/simple/dlib/

1.2	from here use Dlib wheel file based on your Python Version 3.6 I used latest respective version 

 

2)	Scipy : Compute Euclidean Distance between facial landmark points in Eye Aspect Ratio calculation

3)	Playsound Library : to play WAV and MP3 

Pip install Playsound 
Idea of Eye blink and this calculation is based on idea from Paper 
http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf  (Tereza Soukupova and Jan ´ Cech)

How to run : 

Once everything is downloaded  just run python file, i am using argument parser and this is how you this works 
On Command line: 
python .\Sleep_drowsiness_Detector.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav
--shape-predictor : This is the path to dlib’s pre-trained facial landmark detector
-- alarm   path to an input audio file to be used as an alarm, instead of alarm.wav any wav/mp3 file can be used 

Thanks and Happy Learning !!! 




 
