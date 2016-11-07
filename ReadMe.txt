Emotion Recognization:

Training Phase:
-I am training data for each emotion using SVMclassifier (Open Source Code  http://www.gnu.org/licenses/lgpl.html).
-I have used landmark points .txt files captured from each image samples for respective 
 emotions.
-I am calculating displacement of each landmark point from neutral & peak image samples.
-Storing each calculated displacement of x & y landmarak co-ordinates in to csv file.
-Then training svm Classisfier for each emotion.
-Then flowing my test data as stored video frame & live video frame(from web cam) over 
trained data model to get the classification class value.
-Each predition returns 2 values:
 0 - Neutral
 1- Emotion(Happy,Surprised,Sad,Anger)
- I am displaying these values & corresponding emotion on each caputured frame.

For Predition :
- I am providing queryfeature for each frame to trained model.
- I am calculating query feature based on difference between Initial Feature Vector & Current Feature Vector for each frame.
- So the queryfeature stores the displacement of each landmark point.
- Finally, we feed this query feature to trained model for prediction to classify. 

  

