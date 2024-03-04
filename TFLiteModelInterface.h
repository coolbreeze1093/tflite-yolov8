#pragma once

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

#include "opencv2/opencv.hpp"
namespace mediapipe {
	
	class TFLiteModelInterface
	{
	public:
		struct Detection {
			cv::Rect2d boxe;
			int classId=-1;
			float confidenc = 0.0;
			float *maskInfo=nullptr;
			Detection() :maskInfo(new float[32]) { *maskInfo = { 0 }; }
			~Detection() {
				delete maskInfo;
				maskInfo = nullptr;
			}
			Detection(const Detection& self):Detection()
			{
				if (&self == this)
				{
					return;
				}
				this->boxe = self.boxe;
				this->classId = self.classId;
				this->confidenc = self.confidenc;
				std::memcpy(this->maskInfo, self.maskInfo,sizeof(float)*32);
			}
		};
	public:
		TFLiteModelInterface();
		~TFLiteModelInterface();

		bool init(const std::string& modelPath, const int& threads, const bool& useXnnpack);
		bool assessFrame(const cv::Mat& mat, cv::Mat& outMat);
	private:
		std::unique_ptr<tflite::FlatBufferModel> m_model;
		std::unique_ptr<tflite::Interpreter> m_interpreter; // tflite ½âÊÍÆ÷
		TfLiteDelegate* xnnpack_delegate = nullptr;
		float modelConfidenceThreshold{ 0.25 };
		float modelScoreThreshold{ 0.45 };
		float modelNMSThreshold{ 0.50 };

		std::vector<Detection>m_validDetections;

	};
}
