//
//  svmClassifier.h
//  EmotionRecognition
//
//  Created by Himanshu Kandwal on 5/8/16.
//  Copyright © 2016 Himanshu Kandwal. All rights reserved.
//

#ifndef svmClassifier_h
#define svmClassifier_h

#include <opencv/cv.h>
#include <opencv/highgui.h>     // opencv general include file
#include <opencv/ml.h>	  // opencv machine learning include file

using namespace std;
using namespace cv;

class SvmClassifier {
protected:
    CvSVM* svm;
    
public:
    
    float prediction(Mat mat);
    int svmTrain();
    int read_data_from_csv(const char* filename, Mat data, Mat classes, int n_samples );
    int read_testdata_from_csv(const char* filename, Mat data, Mat classes, int n_samples );
    
};

#endif /* svmClassifier_h */