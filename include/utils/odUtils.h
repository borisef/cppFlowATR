#ifndef _ODUTILS_H_
#define _ODUTILS_H_

#include <iostream>
#include <vector>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

using namespace OD;

OD_CycleOutput* NewOD_CycleOutput(int maxNumOfObjects, int defaultImgID_output = 0);
    
void swap_OD_DetectionItem(OD_DetectionItem* xp, OD_DetectionItem * yp);
  
void bubbleSort_OD_DetectionItem(OD_DetectionItem* arr, int n) ;

void PrintColor(int color_id);

cv::Scalar GetColor2Draw(OD::e_OD_TargetColor color_id);

std::string GetStringInitParams(OD::OD_InitParams ip);

std::string CycleOutput2LogString(OD_CycleOutput* co);

std::string DetectionItem2LogString(OD_DetectionItem di);



#endif