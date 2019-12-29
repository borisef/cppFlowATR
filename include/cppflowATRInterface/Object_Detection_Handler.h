#pragma once


#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATR/InterfaceATR.h>
#include <future>


using namespace OD;

class ObjectDetectionManagerHandler:public ObjectDetectionManager
{
    public: 

        OD_ErrorCode InitObjectDetection(OD_InitParams* input) ;
        OD_ErrorCode  OperateObjectDetection(OD_CycleInput* CycleInput, OD_CycleOutput* CycleOutput) ;
        OD_ErrorCode PrepareOperateObjectDetection(OD_CycleInput* CycleInput, OD_CycleOutput* CycleOutput);// run synchroniusly
        int  PopulateCycleOutput(OD_CycleOutput *cycleOutput);
        bool SaveResultsATRimage(OD_CycleInput* ci,OD_CycleOutput* co, char* imgName, bool show);
        OD_InitParams* getParams();
		void setParams(OD_InitParams* ip);
        bool IsBusy(){return m_isBusy;}
        bool m_isBusy = false;
        std::future<OD_ErrorCode> m_result;
        ObjectDetectionManagerHandler();
        ObjectDetectionManagerHandler(OD_InitParams*   ip);

        ~ObjectDetectionManagerHandler();

        mbInterfaceATR* m_mbATR;
    protected: 
       // OD_CycleInput m_cycleInput;

};