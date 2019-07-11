#include "detector.h"

DnnDetector::DnnDetector(String model, String config,
	int Width, int Height, Scalar mmean, bool sswapRB)
{
	path_to_model = model;
	path_to_config = config;
	inputWidth = Width;
	inputHeight = Height;
	mean = mmean;
	swapRB = sswapRB;
	net = readNetFromCaffe(config, model);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}
Mat DnnDetector::Detect(Mat image)
{
	Mat inputTarget;
	blobFromImage(image, inputTarget, 0.007843, Size(inputWidth, inputHeight),
		(127.5, 127.5, 127.5), swapRB, false);
	net.setInput(inputTarget);
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	Size s = detectionMat.size();
	//cout << s << "\n";
	return detectionMat;
}
