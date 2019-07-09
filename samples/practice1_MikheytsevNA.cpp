#include <iostream>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "filter.h"

using namespace cv;
using namespace std;

const char* cmdAbout = "Sample of OpenCV usage. ";

const char* cmdOptions =
"{ i  image         | <none> | image to process        }"
"{ w  width         | <none> | width for image resize  }"
"{ h  height        | <none> | height for image resize }"
"{ q ? help usage   | <none> | print help message      }";

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
    
    // Load image
    String imgName(parser.get<String>("image"));
	cout << imgName;
	Mat Image = imread(imgName);
    
    // Filter image
	GrayFilter gf;
	Mat GImage = gf.ProcessImage(Image);

	int imgWidth(parser.get<int>("width"));
	int imgHeight(parser.get<int>("height"));
	ResizeFilter rf(imgWidth, imgHeight);
	Mat RImage = rf.ProcessImage(Image);

    // Show image
	imshow("image", Image);
	waitKey();

	imshow("image", GImage);
	waitKey();

	imshow("image", RImage);
	waitKey();
    
    
    return 0;
}
