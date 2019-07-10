#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};
class DnnDetector :Detector
{
private:
	String path_to_model, path_to_config, path_to_labels;
	int inputWidth, inputHeight;
	Scalar mean;
	bool swapRB;
public:
	Net net;
	vector<DetectedObject> Detect(Mat image);
	DnnDetector(String path_to_model, String path_to_config, String path_to_labels,
		int inputWidth, int inputHeight, Scalar mean, bool swapRB);
};
