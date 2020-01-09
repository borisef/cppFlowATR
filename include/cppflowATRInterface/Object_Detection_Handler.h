#pragma once


#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATR/InterfaceATR.h>
#include <future>
#include <mutex>


using namespace OD;

OD_CycleInput* NewCopyCycleInput(OD_CycleInput*, uint);
OD_CycleInput* SafeNewCopyCycleInput(OD_CycleInput* ,uint) ;
void DeleteCycleInput(OD_CycleInput* );

class ObjectDetectionManagerHandler:public ObjectDetectionManager
{
    public: 

        OD_ErrorCode InitObjectDetection(OD_InitParams* input) ;
        OD_ErrorCode  OperateObjectDetection(OD_CycleOutput* CycleOutput) ;
        OD_ErrorCode PrepareOperateObjectDetection(OD_CycleInput* CycleInput);// run synchroniusly
        int  PopulateCycleOutput(OD_CycleOutput *cycleOutput);
        bool SaveResultsATRimage(OD_CycleOutput* co, char* imgName, bool show);
        OD_InitParams* getParams();
		void setParams(OD_InitParams* ip);
        bool IsBusy();
        void IdleRun();
        bool GetTargetFromGzir(const char* imgName);
       
        std::future<OD_ErrorCode> m_result;
       
        ObjectDetectionManagerHandler();
        ObjectDetectionManagerHandler(OD_InitParams*   ip);
        bool WaitUntilForThread(int sec);//not in use 
        bool WaitForThread();

        ~ObjectDetectionManagerHandler();

        mbInterfaceATR* m_mbATR = nullptr;
    protected: 
       void DeleteAllInnerCycleInputs();
       OD_CycleInput* m_prevCycleInput = nullptr;
       OD_CycleInput* m_curCycleInput = nullptr;
       OD_CycleInput* m_nextCycleInput = nullptr;
       uint m_numImgPixels = 0;
       uint m_numPtrPixels = 0; 

       std::mutex m_mutexOnNext;
       std::mutex m_mutexOnPrev;

};

