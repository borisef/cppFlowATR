#pragma once
#ifndef NO_TRT
#include "loguru.hpp"
#include <NvInfer.h>

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
#endif // #ifndef NO_TRT
