#include "filter.h"

using namespace cv;
using namespace std;

Mat GrayFilter::ProcessImage(Mat Image)
{
	cvtColor(Image, Image, COLOR_BGR2GRAY);
	return Image;
}

ResizeFilter::ResizeFilter(int newWidth, int newHeight)
{
	width = newWidth;
	height = newHeight;
}
Mat ResizeFilter::ProcessImage(Mat image)
{
	resize(image, image, Size(width, height));
	return image;
}