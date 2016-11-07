
#include "D:/Projects/VisualStudio_2013/ObjectDetection/ObjectDetection/svmClassifier.h"
#include "opencv/cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "dirent.h"
#include <stdio.h>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <iterator>
#include "stasm_lib.h"


using namespace cv;
using namespace std;

SvmClassifier svmclassifier;

// Create a new Haar classifier
static CvHaarClassifierCascade* cascade = 0;

// Function prototype for detecting and drawing an object from an image
void faceDetect(Mat frame, CascadeClassifier cascade_face);

// Create a string that contains the cascade name
//const char* cascade_name = "haarcascade_frontalface_alt2.xml";

const char* path = "frame.jpeg";

int initCount = 0;

float xInit = 0;
float yInit = 0;


float initialFV[154];
float currentFV[154];

string OPENCV_ROOT = "C:/Users/jk/Downloads/opencv/";
//"C:/Ceemple/OpenCV4VS/";
string cascades = OPENCV_ROOT + "sources/data/haarcascades/";
//"samples/data/haarcascades/";
string FACES_CASCADE_NAME = cascades + "haarcascade_frontalface_alt.xml";
string EYES_CASCADE_NAME = cascades + "haarcascade_eye.xml";

void drawEllipse(Mat frame, const Rect rect, int r, int g, int b)
{
	int width2 = rect.width / 2;
	int height2 = rect.height / 2;
	Point center(rect.x + width2, rect.y + height2);
	ellipse(frame, center, Size(width2, height2), 0, 0, 360, Scalar(r, g, b), 2, 8, 0);


}
void rumASM() {
	bool isInitial = false;

	//float xInterim = 0;
	//float yInterim = 0;

	int changed = 0;

	cv::Mat_<unsigned char> img(cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE));

	if (initCount == 6 || (xInit == 0 && yInit == 0))
		isInitial = true;

	if (!img.data)
	{
		printf("Cannot load %s\n", path);
		exit(1);
	}

	int foundface;
	float landmarks[2 * stasm_NLANDMARKS]; // x,y coords (note the 2)

	if (!stasm_search_single(&foundface, landmarks,(const char*)img.data, img.cols, img.rows, path, "C:/Users/jk/Downloads/stasm4.1.0/stasm4.1.0/data"))
	{
		printf("Error in stasm_search_single: %s\n", stasm_lasterr());
		exit(1);
	}
	//cout << foundface << endl;
	if (!foundface) {
		printf("No face found in %s\n", path);
	}
	else
	{
		int index = 0;

		float flowPreviousXPoint = 0;
		float flowPreviousYPoint = 0;
		//float Nsum = 0;
		// draw the landmarks on the image as white dots (image is monochrome)
		stasm_force_points_into_image(landmarks, img.cols, img.rows);
		for (int i = 0; i < stasm_NLANDMARKS; i++) {
			img(cvRound(landmarks[i * 2 + 1]), cvRound(landmarks[i * 2])) = 0;

			if (isInitial == true) {
				initialFV[index++] = cvRound(landmarks[i * 2 + 1]);
				initialFV[index++] = cvRound(landmarks[i * 2]);

				xInit += cvRound(landmarks[i * 2 + 1]);
				yInit += cvRound(landmarks[i * 2]);

			}
			else {

				int frameDifference = initialFV[index] - cvRound(landmarks[i * 2 + 1]);
				
				currentFV[index++] = frameDifference;


				frameDifference = initialFV[index] - cvRound(landmarks[i * 2]);
				currentFV[index++] = frameDifference;

			}
		}

		cv::Mat queryFeature = cv::Mat(1, 154, CV_32F, currentFV);
		
		float isHappy = svmclassifier.predictionHappy(queryFeature);
		float isSurprised = svmclassifier.predictionSurprised(queryFeature);
		float isDisgust = svmclassifier.predictionDisgust(queryFeature);
		float isAnger = svmclassifier.predictionAnger(queryFeature);

	//	cout << isHappy << " - " << isSurprised << " - " << isDisgust << " - " << isAnger << endl;

		string predictionInformation;

		predictionInformation = " Neutral ";
		if (isHappy != 0)
			predictionInformation = " HAPPY ";
		else
		if (isAnger != 0)
			predictionInformation = " ANGER ";
		else if (isSurprised != 0)
			predictionInformation = " SURPRISED ";
		else if (isDisgust != 0)
			predictionInformation = " SAD ";
		


     	cout << predictionInformation << endl;
		putText(img, predictionInformation, Point(10, 10), CV_FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1);

	}
	

	cv::imwrite("minimal.bmp", img);

	//cv::resize(img, img, Size(800, 800), 0, 0, INTER_CUBIC);

	cvNamedWindow("Video Frame", CV_WINDOW_NORMAL);
	cv::imshow("Video Frame", img);

}

// Function to detect and draw any faces that is present in an image
void faceDetect(Mat frame, CascadeClassifier cascade_face)
{
	Mat frame_gray;
	vector<Rect> faces;



	cvtColor(frame, frame_gray, CV_BGR2GRAY);


	// equalizeHist(frame_gray, frame_gray); // input, outuput
	//  medianBlur(frame_gray, frame_gray, 5); // input, output, neighborhood_size
	//  blur(frame_gray, frame_gray, Size(5,5), Point(-1,-1));
	/*  input,output,neighborood_size,center_location (neg means - true center) */


	cascade_face.detectMultiScale(frame_gray, faces,
		1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	/* frame_gray - the input image
	faces - the output detections.
	1.1 - scale factor for increasing/decreasing image or pattern resolution
	3 - minNeighbors.
	larger (4) would be more selective in determining detection
	smaller (2,1) less selective in determining detection
	0 - return all detections.
	0|CV_HAAR_SCALE_IMAGE - flags. This flag means scale image to match pattern
	Size(30, 30)) - size in pixels of smallest allowed detection
	*/
	Mat faceROI;
	int detected = 0;

	int nfaces = (int)faces.size();
	for (int i = 0; i < nfaces; i++)
	{
		Rect face = faces[i];
		drawEllipse(frame, face, 255, 0, 255);
	    faceROI = frame_gray(face);
		
	}
	// --- stasm code
	

	imwrite(path, faceROI);

	rumASM();
}

int main()
{
	svmclassifier.svmTrain();

	std::string filename = "D:/Projects/VisualStudio_2013/DetectWinkProject/1.avi";
	//std::string filename =  "C:/Users/jk/Downloads/EmotionRecognitionVideo2.mov";

	bool result = true;

	CascadeClassifier faces_cascade, eyes_cascade;

	if (!faces_cascade.load(FACES_CASCADE_NAME) || !eyes_cascade.load(EYES_CASCADE_NAME))
	{
		cerr << FACES_CASCADE_NAME << " or " << EYES_CASCADE_NAME
			<< " are not in a proper cascade format" << endl;
		waitKey(0);
		return(-1);
	}
	bool webcam = true;

	if (!webcam)
	{

		VideoCapture capture;

		// Load the image from that filename
		if (filename.length() == 0) {
			capture.open(0);
		}
		else {
			capture.open(filename);
		}

		Mat frame;

		if (capture.isOpened())
		{
			while(1) {
				long loop = 0;

				if (loop++ % 30 == 0) {


					capture >> frame;

					if (!frame.empty()) {
						IplImage* image = cvCreateImage(cvSize(frame.cols, frame.rows), 8, 3);
						IplImage ipltemp = frame;
						cvCopy(&ipltemp, image);

						faceDetect(frame, faces_cascade);
					}
					else {
						printf(" --(!) No captured frame -- Break!"); break;
					}

					int c = cvWaitKey(1);

					if (char(c) == 27)
						break;
				}
			}
		}
	}
	else
	{
		VideoCapture videocapture(0);
		if (!videocapture.isOpened()) {
			cerr << "Can't open default video camera" << endl;
			exit(1);
		}
		string windowName = "Live Video";
		namedWindow("video", CV_WINDOW_AUTOSIZE);
		Mat frame;
		bool finish = false;
		while (!finish) {
			if (!videocapture.read(frame)) {
				cout << "Can't capture frame" << endl;
				break;
			}
			faceDetect(frame, faces_cascade);
			imshow("video", frame);
			if (waitKey(30) >= 0) finish = true;
		}
	}

	// Destroy the window previously created with filename: "result"
	cvDestroyWindow("result");

	return 0;
}