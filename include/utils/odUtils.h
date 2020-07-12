#ifndef _ODUTILS_H_
#define _ODUTILS_H_

#include <iostream>
#include <vector>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <map>

using namespace OD;

enum ATR_TargetSubClass_MB //BE
{
	ATR_PERSON			= 1,
	ATR_CAR				= 5,
	ATR_BICYCLE			= 10,
	ATR_MOTORCYCLE		= 11,
	ATR_BUS				= 12,
	ATR_TRUCK			= 13,
	ATR_VAN				= 14,
	ATR_JEEP 			= 7,
    ATR_PICKUP       	= 19,
	ATR_PICKUP_OPEN 	= 15,
	ATR_PICKUP_CLOSED 	= 16,
	ATR_FORKLIFT 		= 17,
	ATR_TRACKTOR 		= 18,
	ATR_STATION 		= 6,
	ATR_OTHER			= 999
};


OD_CycleOutput* NewOD_CycleOutput(int maxNumOfObjects, int defaultImgID_output = 0);
    
void swap_OD_DetectionItem(OD_DetectionItem* xp, OD_DetectionItem * yp);
  
void bubbleSort_OD_DetectionItem(OD_DetectionItem* arr, int n) ;

void PrintColor(int color_id);

cv::Scalar GetColor2Draw(OD::e_OD_TargetColor color_id);

std::string GetColorString(const OD::e_OD_TargetColor color_id);

std::string GetStringInitParams(OD::OD_InitParams ip);

std::string CycleOutput2LogString(OD_CycleOutput* co);

std::string DetectionItem2LogString(OD_DetectionItem di);

std::string GetFromMapOfClasses(e_OD_TargetClass cl);

std::string GetFromMapOfSubClasses(e_OD_TargetSubClass scl);

void MapATR_Classes(ATR_TargetSubClass_MB inClass, OD::e_OD_TargetClass& outClass, OD::e_OD_TargetSubClass& outSubClass);

int FilterCycleOutputByClassNoSqueeze(OD_CycleOutput* co, e_OD_TargetClass class2remove);

int FilterCycleOutputBySubClassNoSqueeze(OD_CycleOutput* co, e_OD_TargetSubClass subclass2remove);

int SqueezeCycleOutputInplace(OD_CycleOutput* co);

float ValidityScore(MB_Mission mbMission, OD_DetectionItem* detItem, int cyclesNotConfirmed, float currentScore);

float ValidityScoreColor(e_OD_TargetColor tcolor, e_OD_TargetColor detectedColor);

float ValidityScoreClass(e_OD_TargetSubClass tclass, e_OD_TargetSubClass detectedClass);


#endif