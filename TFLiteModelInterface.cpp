#include "TFLiteModelInterface.h"
#include <iostream>
#include "BeautyMatting/glogInterface/GlogInterface.h"
#include <opencv2/dnn.hpp>
#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fstream>

using namespace mediapipe;
TFLiteModelInterface::TFLiteModelInterface()
{

}

TFLiteModelInterface::~TFLiteModelInterface()
{
	if (xnnpack_delegate != nullptr)
	{
		TfLiteXNNPackDelegateDelete(xnnpack_delegate);
	}
}

bool TFLiteModelInterface::init(const std::string& modelPath, const int& threads, const bool& useXnnpack)
{
	// 1，创建模型和解释器对象，并加载模型
	m_model = tflite::FlatBufferModel::BuildFromFile(modelPath.c_str());
	if (!m_model) {
		SIM_LOG_ERROR << "\nFailed to mmap model " << modelPath << "\n";
		return false;
	}
	SIM_LOG_INFO << "Loaded model " << modelPath << "\n";

	// 2，将模型中的tensor映射写入到解释器对象中
	tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
	tflite::InterpreterBuilder interpreter_builder(*m_model, resolver);

	/*TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
	options.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
	options.serialization_dir = kTmpDir;
	options.model_token = kModelToken;

	auto* delegate = tflite::TfLiteGpuDelegateV2Create(options);
	if (m_interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;*/

	if (useXnnpack)
	{
		TfLiteXNNPackDelegateOptions xnnpack_options = TfLiteXNNPackDelegateOptionsDefault();
		xnnpack_options.num_threads = 20;

		xnnpack_delegate = TfLiteXNNPackDelegateCreate(&xnnpack_options);
		if (xnnpack_delegate == nullptr) {
			std::cout << "Failed to construct interpreter\n";
			return false;
		}

		interpreter_builder.AddDelegate(xnnpack_delegate);
	}

	interpreter_builder.SetNumThreads(threads);

	if (interpreter_builder(&m_interpreter) != kTfLiteOk)
	{
		SIM_LOG_ERROR << "Failed to construct interpreter\n";

		return false;
	}

	if (m_interpreter == nullptr)
	{
		SIM_LOG_ERROR << "interpreter is empty\n";
		return false;
	}

	if (m_interpreter->AllocateTensors() != kTfLiteOk) {
		SIM_LOG_ERROR << "Failed to allocate tensors!";
		return false;
	}

	return true;
}



bool TFLiteModelInterface::assessFrame(const cv::Mat& mat, cv::Mat& outMat)
{
	clock_t _preProcessStart, _preProcessEnd,_startAll,_endAll;
	_startAll = clock();
	_preProcessStart = clock();
	m_validDetections.clear();

	if (!m_model) {
		SIM_LOG_ERROR << "Failed to mmap model\n";
		return false;
	}
	if (mat.empty())
	{
		SIM_LOG_ERROR << "input mat is empty\n";
		return false;
	}
	//输入图片转张量
	const std::vector<int> input_tensors = m_interpreter->inputs();
	if (input_tensors.size() <= 0)
	{
		SIM_LOG_ERROR << "input tensors is empty\n";
		return false;
	}
	int _input0 = input_tensors.at(0);
	TfLiteIntArray* _inputDims = m_interpreter->tensor(_input0)->dims;
	std::vector<int>_inputdimVector;
	for (int i = 0; i < _inputDims->size; ++i)
	{
		_inputdimVector.push_back(_inputDims->data[i]);
	}
	if (_inputdimVector.size() < 4)
	{
		SIM_LOG_ERROR << "input tensors dims is error\n";
		return false;
	}

	int _orgWidth = mat.cols;
	int _orgHeight = mat.rows;

	float x_factor = _orgWidth / (float)_inputdimVector.at(1);
	float y_factor = _orgHeight / (float)_inputdimVector.at(2);

	const cv::RotatedRect rotated_rect(cv::Point2f(_orgWidth / 2.0, _orgHeight / 2.0),
		cv::Size2f(_orgWidth, _orgHeight),
		0 * 180.f / 3.1415926);
	cv::Mat src_points;
	cv::boxPoints(rotated_rect, src_points);
	/* clang-format off */
	float dst_corners[8] = { 0.0f,      _inputdimVector.at(2),
							0.0f,      0.0f,
							_inputdimVector.at(1), 0.0f,
							_inputdimVector.at(1), _inputdimVector.at(2) };
	/* clang-format on */
	cv::Mat dst_points = cv::Mat(4, 2, CV_32F, dst_corners);
	cv::Mat projection_matrix =
		cv::getPerspectiveTransform(src_points, dst_points);
	cv::Mat transformed;
	cv::warpPerspective(mat, transformed, projection_matrix,
		cv::Size(_inputdimVector.at(1), _inputdimVector.at(2)),
		/*flags=*/cv::INTER_LINEAR,
		/*borderMode=*/cv::BORDER_CONSTANT);

	cv::Mat input_im_32F(_inputdimVector.at(2), _inputdimVector.at(1), CV_32FC3);
	cv::cvtColor(transformed, transformed, cv::COLOR_BGR2RGB);
	transformed.convertTo(input_im_32F, CV_32FC3, 1.0 / 255.0, 0.0);

	Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> _input_im_32F_tensor_data((float*)input_im_32F.data, 
		_inputdimVector.at(0), _inputdimVector.at(1), _inputdimVector.at(2), _inputdimVector.at(3));

	//张量加到模型
	float * _modelInput = m_interpreter->typed_tensor<float>(_input0);
	std::memcpy(_modelInput, _input_im_32F_tensor_data.data(), _input_im_32F_tensor_data.size() * sizeof(float));
	_preProcessEnd = clock();
	SIM_LOG_INFO <<"preProcess time(ms): " << int(_preProcessEnd - _preProcessStart);

	clock_t _startInvoke, _endInvoke;
	_startInvoke = clock();
	//推理
	if (m_interpreter->Invoke() != kTfLiteOk) {
		SIM_LOG_ERROR << "invoke error";
		return false;
	}
	_endInvoke = clock();
	SIM_LOG_INFO <<"Invoke  time(ms): " << int(_endInvoke - _startInvoke);

	std::vector<cv::Mat> _maskVect;
	//张量转mask
	for (int tensor_index : m_interpreter->outputs()) {
		if (m_interpreter->EnsureTensorDataIsReadable(tensor_index) == kTfLiteOk)
		{
			
			TfLiteIntArray * _outputDims = m_interpreter->tensor(tensor_index)->dims;
			std::vector<int>_outdimVector;
			for (int i=0; i<_outputDims->size;++i)
			{
				_outdimVector.push_back(_outputDims->data[i]);
			}
			
			switch (m_interpreter->tensor(tensor_index)->type)
			{
			case kTfLiteNoType:
				break;
			case kTfLiteFloat32:
			{
				std::vector<Detection>_detections;
				float* _output = m_interpreter->typed_tensor<float>(tensor_index);
				if (_outdimVector.size() == 3)
				{
					Eigen::TensorMap<Eigen::Tensor<float, 3, Eigen::RowMajor>> tensor(_output, _outdimVector.at(0), _outdimVector.at(1), _outdimVector.at(2));
					Eigen::Tensor<float, 2, Eigen::RowMajor> matrix = tensor.chip<0>(0);
					Eigen::array<int, 2> shuffle_order = { 1, 0 };
					Eigen::Tensor<float, 2, Eigen::RowMajor> shuffled_tensor = matrix.shuffle(shuffle_order);
					float* _newTensor = shuffled_tensor.data();

					std::vector<cv::Rect2d> _boxes;
					std::vector<float> _confidences;
					
					for (size_t dim2 = 0; dim2 < _outdimVector.at(2); dim2++)
					{
						size_t _curPtrPose = dim2 * _outdimVector.at(1);
						std::vector<float> _classes(80);
						std::memcpy(_classes.data(), _newTensor + _curPtrPose + 4, sizeof(float) * 80);
						auto _maxIt = std::max_element(_classes.begin(), _classes.end());

						if (*_maxIt < modelConfidenceThreshold)
						{
							continue;
						}

						float _x = *(_newTensor + _curPtrPose + 0);
						float _y = *(_newTensor + _curPtrPose + 1);
						float _w = *(_newTensor + _curPtrPose + 2);
						float _h = *(_newTensor + _curPtrPose + 3);

						Detection _d;
						_d.boxe = cv::Rect2d(_x, _y, _w, _h);
						_d.confidenc = *_maxIt;
						int _maxIndex = std::distance(_classes.begin(), _maxIt);
						_d.classId = _maxIndex;

						std::memcpy(_d.maskInfo, _newTensor + _curPtrPose + 84, sizeof(float) * 32);

						_boxes.push_back(_d.boxe);
						_confidences.push_back(_d.confidenc);

						_detections.push_back(std::move(_d));
					}
					

					std::vector<int> nms_result;
					cv::dnn::NMSBoxes(_boxes, _confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

					for (auto var : nms_result)
					{
						m_validDetections.push_back(_detections.at(var));
					}
				}
				else if (_outdimVector.size() == 4)
				{

					Eigen::TensorMap<Eigen::Tensor<float, 4, Eigen::RowMajor>> tensor(_output, _outdimVector.at(0), _outdimVector.at(1), _outdimVector.at(2), _outdimVector.at(3));
					Eigen::Tensor<float, 3, Eigen::RowMajor> matrix = tensor.chip<0>(0);

					Eigen::array<int, 3> shuffle_order = { 2, 0, 1 };
					Eigen::Tensor<float, 3, Eigen::RowMajor> shuffled_tensor = matrix.shuffle(shuffle_order);
				
					Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> reshaped(shuffled_tensor.data(), _outdimVector.at(3), _outdimVector.at(1) * _outdimVector.at(2));
					int _outMatIndex=0;
					for (auto var : m_validDetections)
					{
						_outMatIndex++;
						Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> maskInfo(var.maskInfo, 1,32);
						// 定义乘法操作的维度
						Eigen::array<Eigen::IndexPair<int>, 1> product_dims = { Eigen::IndexPair<int>(1, 0) };
						// 进行矩阵乘法
						Eigen::Tensor<float, 2, Eigen::RowMajor> rMask = maskInfo.contract(reshaped, product_dims);

						Eigen::TensorMap<Eigen::Tensor<float, 2, Eigen::RowMajor>> rMaskReshaped(rMask.data(), _outdimVector.at(1), _outdimVector.at(2));

						cv::Mat _mat(160, 160, CV_32FC1);
						std::memcpy(_mat.data, rMask.data(), 160 * 160 * sizeof(float));
						cv::threshold(_mat, _mat, 0.5, 255, cv::THRESH_BINARY);
						cv::resize(_mat, _mat, cv::Size(_orgWidth, _orgHeight));

						/*int left = int((_x - 0.5 * _w) * x_factor);
						int top = int((_y - 0.5 * _h) * y_factor);

						int width = int(_w * x_factor);
						int height = int(_h * y_factor);

						cv::imwrite("./"+std::to_string(_outMatIndex) + ".png", _mat);*/
					}
				}
			}
				break;
			case kTfLiteInt32:
				break;
			case kTfLiteUInt8:
				break;
			case kTfLiteInt64:
				break;
			case kTfLiteString:
				break;
			case kTfLiteBool:
				break;
			case kTfLiteInt16:
				break;
			case kTfLiteComplex64:
				break;
			case kTfLiteInt8:
				break;
			case kTfLiteFloat16:
				break;
			case kTfLiteFloat64:
				break;
			case kTfLiteComplex128:
				break;
			case kTfLiteUInt64:
				break;
			case kTfLiteResource:
				break;
			case kTfLiteVariant:
				break;
			case kTfLiteUInt32:
			default:
				break;
			}
		}
		else
		{
			SIM_LOG_ERROR << "Tensor is empty";
			return false;
		}
	}

	if (outMat.empty())
	{
		SIM_LOG_ERROR << "output mat is empty";
		return false;
	}
	_endAll = clock();
	SIM_LOG_INFO << "All time(ms): " << int(_endAll - _startAll);

	return true;
}
