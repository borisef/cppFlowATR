#pragma once
#ifndef NO_TRT
#include <string>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>

#include <cppflowATRInterface/Object_Detection_Types.h>

#include "utils/trt_loguru_wrapper.h"
#include "utils/trt_buffers.h"
#include "utils/loguru.hpp"

#include "InterfaceCMbase.h"

class mbInterfaceCMTrt : public mbInterfaceCMbase
{
protected:
  TRTLoguruWrapper m_logger;
  nvinfer1::ICudaEngine* m_engine;
  nvinfer1::IExecutionContext* m_context;
  cudaStream_t m_stream;
  cudaEvent_t m_start;
  cudaEvent_t m_end;
  BufferManager *m_bufferManager;
  int m_DLACore;
  bool m_isInitialized;
  std::map<int, std::string> m_inTensors;
  std::map<int, std::string> m_outTensors;

public:
  //constructors/destructors
  mbInterfaceCMTrt();
  mbInterfaceCMTrt(int h, int w, int nc, int bs, bool hbs);
  virtual ~mbInterfaceCMTrt();

  void initTrtResources();
  bool isInitialized() { return m_isInitialized; };
  bool doInference();
  bool normalizeImageIntoInputBuffer(const cv::Mat &img);
  bool normalizeImagesIntoInputBuffer(const std::vector<cv::Mat> &images);
  bool LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor);
  std::vector<float> RunRGBimage(cv::Mat img);
  std::vector<std::vector<float>> RunRGBimagesBatch(const std::vector<cv::Mat> &images);
  std::vector<float> RunRGBImgPath(const unsigned char *ptr);
  std::vector<float> RunImgBB(cv::Mat img, OD::OD_BoundingBox bb);
  bool RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults = true);
  void IdleRun();
};
#endif //#ifndef NO_TRT