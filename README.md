# Age-Detection-using-image-processing
Age detector that can approximately detects the gender and age of the person (face) using image processing and  Deep Learning.

Objective :
To build an age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.

About the Project :
In this Python Project, I have employed techniques of image processing and  Deep Learning to accurately identify the gender and age of a person using webcam. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 3), (3 – 7), (7 – 14), (14 – 22), (22 – 35), (35 – 44), (44 – 57), (57 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age using the webcam because of factors like makeup, lighting, obstructions, and facial expressions. Thus, I have made this a classification problem instead of making it one of regression.

Additional Python Libraries Required :
OpenCV
   pip install opencv-python
argparse
   pip install argparse
Numpy
   pip install numpy
Matplotlib
   pip install matplotlib
   
The contents of this Project :

age.caffemodel,
age.prototxt,
face_detector.pb,
face_detector.pbtxt,
gender.caffemodel,
gender.prototxt,
Age detection through image processing.ipynb

For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.

Example :

Gender: Male
Age: 14-22 years
