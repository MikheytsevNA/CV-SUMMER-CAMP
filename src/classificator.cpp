#include "classificator.h"

DnnClassificator::DnnClassificator(String path_to_model, String path_to_config, String path_to_labels,
	int inputWidth, int inputHeight, Scalar mmean, bool sswapRB)
{
	model = path_to_model;
	config = path_to_config;
	labels = path_to_labels;
	Width = inputWidth;
	Height = inputHeight;
	mean = mmean;
	swapRB = sswapRB;
	net = readNet(model, config);
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);
}
Mat DnnClassificator::Classify(Mat image)
{
	Mat inputTensor;
	blobFromImage(image, inputTensor, 1.0, Size(Width, Height),
		mean, swapRB, false);
	net.setInput(inputTensor);
	net.forward();
	return inputTensor.reshape(1, 1);
}