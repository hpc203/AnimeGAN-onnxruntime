#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class AnimeGAN
{
public:
	AnimeGAN();
	Mat detect(Mat& cv_image);
private:
	const int inpWidth = 512;
	const int inpHeight = 512;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "AnimeGANv2");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};

AnimeGAN::AnimeGAN()
{
	string model_path = "face_paint_512_v2_0.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
}

Mat AnimeGAN::detect(Mat& srcimg)
{
	const int height = srcimg.rows;
	const int width = srcimg.cols;
	Mat img;
	resize(srcimg, img, Size(this->inpWidth, this->inpHeight));
	this->input_image_.resize(this->inpHeight * this->inpWidth * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];   ///BGR2RGB
				this->input_image_[c * this->inpHeight * this->inpWidth + i * this->inpWidth + j] = pix * 2 - 1;
			}
		}
	}
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
	float* preds = ort_outputs[0].GetTensorMutableData<float>();
	Mat dst = img.clone();
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < this->inpHeight; i++)
		{
			for (int j = 0; j < this->inpWidth; j++)
			{
				float pix = preds[0] * 0.5 + 0.5;
				pix = (pix > 0) ? pix : 0;
				pix = (pix < 1) ? pix : 1;
				pix *= 255;
				dst.ptr<uchar>(i)[j * 3 + 2 - c] = (uchar)pix;
				preds++;
			}
		}
	}
	resize(dst, dst, Size(width, height));
	return dst;
}

int main()
{
	AnimeGAN mynet;   
	string imgpath = "liushishi.jpg";
	Mat srcimg = imread(imgpath);
	Mat dstimg = mynet.detect(srcimg);

	
	static const string kWinName = "Deep learning AnimeGAN in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, dstimg);
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", srcimg);
	waitKey(0);
	destroyAllWindows();
}