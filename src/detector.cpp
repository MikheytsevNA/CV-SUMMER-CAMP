#include "detector.h"

DnnDetector::DnnDetector(String model, String config, String labels,
	int Width, int Height, Scalar mmean, bool sswapRB)
{
	path_to_model = model;
	path_to_config = config;
	path_to_labels = labels;
	inputWidth = Width;
	inputHeight = Height;
	mean = mmean;
	swapRB = sswapRB;
	net = readNet(model, config);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
}
vector<DetectedObject> DnnDetector::Detect(Mat image)
{
	Mat inputTarget;
	blobFromImage(image, inputTarget, 1.0, Size(inputWidth, inputHeight),
		mean, swapRB, false);
	net.setInput(inputTarget);
	vector<DetectedObject> result = net.forward();
	return result;
}
