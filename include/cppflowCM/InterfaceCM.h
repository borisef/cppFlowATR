#pragma once

#include <string>

#include <opencv2/opencv.hpp>
#include <cppflow/Tensor.h>
#include <cppflow/Model.h>
#include <cppflowATRInterface/Object_Detection_Types.h>

#include "InterfaceCMbase.h"

class mbInterfaceCM : public mbInterfaceCMbase
{
protected:
  Model *m_model;
  Tensor *m_inTensorPatches;
  Tensor *m_outTensorScores;

  // int m_patchHeight = 128;
  // int m_patchWidth = 128;
  // int m_batchSize = 32;
  // int m_numColors = 7;
  // bool m_hardBatchSize = false;
  //float m_tileMargin = 0.2;

public:
  //constructors/destructors
  mbInterfaceCM();
  mbInterfaceCM(int h, int w, int nc, int bs, bool hbs);
  virtual ~mbInterfaceCM();

  bool LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor);
  std::vector<float> RunRGBimage(cv::Mat img);
  std::vector<float> RunRGBImgPath(const unsigned char *ptr);
  std::vector<float> RunImgBB(cv::Mat img, OD::OD_BoundingBox bb);
  bool RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults = true);
  OD::e_OD_TargetColor TargetColor(uint cid);
  void IdleRun();
};