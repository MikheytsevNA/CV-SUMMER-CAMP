#pragma once
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class Classificator
{
public:
    vector<string> classesNames;
    virtual Mat Classify(Mat image) = 0 {}
};
class DnnClassificator:Classificator
{
private:
	String model, config, labels;
	int Width, Height;
	Scalar mean;
	bool swapRB;
public:
	DnnClassificator(String path_to_model, String path_to_config, String path_to_labels,
		int inputWidth, int inputHeight, Scalar mean, bool swapRB);
	Net net;
	Mat Classify(Mat image);
};