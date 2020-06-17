#pragma once

#include <string>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <opencv2/opencv.hpp>

#include <cppflowATRInterface/Object_Detection_Types.h>

#include "utils/trt_buffers.h"
#include "utils/loguru.hpp"

#include "InterfaceCMbase.h"

class TRTLoguruWrapper : public nvinfer1::ILogger
{
public:
  TRTLoguruWrapper() { }

  loguru::NamedVerbosity getVerbosity(nvinfer1::ILogger::Severity severity) const
  {
    static loguru::NamedVerbosity SevirityToVerbosityMapping[] = {
        loguru::Verbosity_INVALID, // kINTERNAL_ERROR = 0
        loguru::Verbosity_ERROR,   // kERROR = 1
        loguru::Verbosity_WARNING, // kWARNING = 2
        loguru::Verbosity_INFO,    // kINFO = 3
        loguru::Verbosity_1,       // kVERBOSE = 4
    };
    return SevirityToVerbosityMapping[(int)severity];
  }

  void log(nvinfer1::ILogger::Severity severity, const char *msg)
  {
    loguru::NamedVerbosity verbosity = getVerbosity(severity);
    VLOG_F(verbosity, msg);
  }
};

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
  bool LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor);
  std::vector<float> RunRGBimage(cv::Mat img);
  std::vector<float> RunRGBImgPath(const unsigned char *ptr);
  std::vector<float> RunImgBB(cv::Mat img, OD::OD_BoundingBox bb);
  bool RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults = true);
  void IdleRun();
};
