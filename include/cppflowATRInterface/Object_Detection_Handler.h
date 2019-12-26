#pragma once


#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATR/InterfaceATR.h>


using namespace OD;

class ObjectDetectionManagerHandler:public ObjectDetectionManager
{
    public: 

        OD_ErrorCode InitObjectDetection(OD_InitParams* input) ;
        OD_ErrorCode  OperateObjectDetection(OD_CycleInput* CycleInput, OD_CycleOutput* CycleOutput) ;
        int  PopulateCycleOutput(OD_CycleOutput *cycleOutput);
        bool SaveResultsATRimage(OD_CycleInput* ci,OD_CycleOutput* co, char* imgName, bool show);
        OD_InitParams* getParams();
		void setParams(OD_InitParams* ip);
        bool m_isBusy = false;
        ObjectDetectionManagerHandler();
        ObjectDetectionManagerHandler(OD_InitParams*   ip);

        ~ObjectDetectionManagerHandler();

        mbInterfaceATR* m_mbATR;

};
