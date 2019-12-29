///////////////////////////////////////////////////////////////////////////
// File Name:   Object_Detection_API.h
//
// Description: An API for Object Detection algorithem
// 
// Copyright (c) 2019  Rafael Ltd. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef OBJECT_DETECTION_API_H
#define OBJECT_DETECTION_API_H

//#include <cppflowATR/InterfaceATR.h>

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

//class DECLARE_API_FUNCTION ObjectDetectionManager;

#ifdef WIN32
    class  ObjectDetectionManager // for windows
#else
class DECLARE_API_FUNCTION ObjectDetectionManager
#endif
{	protected:
		OD_InitParams* m_initParams;

	public:
		virtual OD_ErrorCode InitObjectDetection( OD_InitParams* input) = 0 ;
		virtual OD_ErrorCode  OperateObjectDetection( OD_CycleInput* CycleInput, OD_CycleOutput* CycleOutput) = 0;
		
		
		bool SaveResultsATRimage(OD_CycleInput* ci,OD_CycleOutput* co, char* imgName, bool show);
		
		//constructors
		ObjectDetectionManager(OD_InitParams* ip);
		ObjectDetectionManager();

};


extern "C" 
{
	DECLARE_API_FUNCTION  ObjectDetectionManager* CreateObjectDetector(OD_InitParams *);
	DECLARE_API_FUNCTION  OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager*);
	DECLARE_API_FUNCTION  OD_ErrorCode InitObjectDetection(ObjectDetectionManager*, OD_InitParams *);
	DECLARE_API_FUNCTION  OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager* , OD_CycleInput* , OD_CycleOutput* );
	DECLARE_API_FUNCTION  OD_ErrorCode ResetObjectDetection(ObjectDetectionManager*);
	//DECLARE_API_FUNCTION  OD_ErrorCode DeleteObjectDetection(ObjectDetectionManager*);
	DECLARE_API_FUNCTION  OD_ErrorCode GetMetry(ObjectDetectionManager*, int , void *);
}
}

#endif // OBJECT_DETECTION_API_H


