#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
"{ i  image                             | <none> | image to process                  }"
"{ w  width                             |        | image width for classification    }"
"{ h  height                            |        | image heigth fro classification   }"
"{ model_path                           |        | path to model                     }"
"{ detector_path                        |        | path to model configuration       }"
"{ mean                                 |        | vector of mean model values       }"
"{ swap                                 |        | swap R and B channels. TRUE|FALSE }"
"{ acc                                  |        | Threshhold accuracy for detection }"
"{ q ? help usage                       |        | print help message                }";


int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);

  // If help option is given, print help message and exit.
  if (parser.get<bool>("help")) {
    parser.printMessage();
    return 0;
  }

  // Do something cool.
   //Input arguments
  String imgName(parser.get<String>("image"));
  String model(parser.get<String>("model_path"));
  String config(parser.get<String>("detector_path"));
  Mat image = imread(imgName);
  //namedWindow("Random Picture", WINDOW_NORMAL);
  //imshow("Random Picture", image);
  //waitKey();

  int Width(parser.get<int>("width")), Height(parser.get<int>("height"));
  double acc(parser.get<double>("acc"));
  Scalar mean(parser.get<int>("mean"));
  bool swapRB(parser.get<bool>("swap"));
  DnnDetector DnnNet(model, config, Width, Height, mean, swapRB);
  Mat result = DnnNet.Detect(image);
  for (int i = 0; i < result.rows; i++)
  {
	  float confidence = result.at<float>(i, 2);

	  if (confidence > acc)
	  {
		  int idx = static_cast<int>(result.at<float>(i, 1));
		  int xLeftBottom = static_cast<int>(result.at<float>(i, 3) * image.cols);
		  int yLeftBottom = static_cast<int>(result.at<float>(i, 4) * image.rows);
		  int xRightTop = static_cast<int>(result.at<float>(i, 5) * image.cols);
		  int yRightTop = static_cast<int>(result.at<float>(i, 6) * image.rows);

		  Rect object((int)xLeftBottom, (int)yLeftBottom,
			  (int)(xRightTop - xLeftBottom),
			  (int)(yRightTop - yLeftBottom));

		  rectangle(image, object, Scalar(0, 255, 0), 2);
	  }
  }
  //cout << result.reshape(1,100);
  //cout << result <<"\n";
  //rectangle(image, Point(100, 100), Point(200, 200), Scalar(0, 255, 0));
  imshow("Random Picture", image);
  waitKey();
  
  return 0;
}