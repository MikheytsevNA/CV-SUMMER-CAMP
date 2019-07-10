#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "classificator.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ config_path                          |        | path to model configuration       }"
"{ label_path                           |        | path to class labels              }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ q ? help usage                       |        | print help message                }";

int main(int argc, char** argv)
{
	// Process input arguments
	CommandLineParser parser(argc, argv, cmdOptions);
	parser.about(cmdAbout);

	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}

	// Load image and init parameters
	String imgName(parser.get<String>("image"));
	Mat image = imread(imgName);
	namedWindow("Picture", WINDOW_NORMAL);
	imshow("Picture", image);
	//waitKey();
	String path_to_model(parser.get<String>("model_path"));
	String path_to_config(parser.get<String>("config_path"));
	String path_to_labels(parser.get<String>("label_path"));
	int inputWidth(parser.get<int>("width"));
	int inputHeight(parser.get<int>("height"));
	Scalar mmean(parser.get<Scalar>("mean"));
	bool sswapRB(parser.get<bool>("swap"));


	//Image classification
	DnnClassificator DnnNet(path_to_model, path_to_config, path_to_labels, inputWidth, inputHeight, mmean, sswapRB);
	Mat prob = DnnNet.Classify(image);
	
	//Show result
	Point classIdPoint;
	double confidence;
	minMaxLoc(prob, 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;
	cout << "Class: " << classId << "\n";
	//cout << "Confidence: " << confidence << "\n";

	return 0;
}
