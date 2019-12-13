///////////////////////////////////////////////////////////////////////////
// File Name:   Object_Detection_API.h
//
// Description: An API for Object Detection algorithem
// 
// Copyright (c) 2019  Rafael Ltd. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef OBJECT_DETECTION_API_H
#define OBJECT_DETECTION_API_H

#include <cppflowATR/InterfaceATR.h>

#include "Object_Detection_Types.h"


/////////////////////////////////////////////////////////////////////////////
//Windows DLL / Linux SO
 /////////////////////////////////////////////////////////////////////////////
namespace OD
{
#if defined(__GNUC__)
#define DECLARE_API_FUNCTION __attribute__((visibility("default")))
#else 
#ifdef OBJECT_DETECTION_DLL_IMPLEMENTATION
#define DECLARE_API_FUNCTION  __declspec(dllexport)
#else
#define DECLARE_API_FUNCTION  __declspec(dllimport)
#endif
#endif

#define OD_API_VERSION 0

class DECLARE_API_FUNCTION ObjectDetectionManager
{
	public:
		mbInterfaceATR* m_mbATR;

};


extern "C" 
{
	DECLARE_API_FUNCTION  ObjectDetectionManager* CreateObjectDetector(OD_InitParams *);
	DECLARE_API_FUNCTION  OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager*);
	DECLARE_API_FUNCTION  OD_ErrorCode InitObjectDetection(ObjectDetectionManager*, OD_InitParams *);
	DECLARE_API_FUNCTION  OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager* , OD_CycleInput* , OD_CycleOutput* );
	DECLARE_API_FUNCTION  OD_ErrorCode ResetObjectDetection(ObjectDetectionManager*);
	DECLARE_API_FUNCTION  OD_ErrorCode GetMetry(ObjectDetectionManager*, int size, void *metry);
}
}

#endif // OBJECT_DETECTION_API_H


