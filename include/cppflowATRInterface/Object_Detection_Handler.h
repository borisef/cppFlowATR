#pragma once

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATR/InterfaceATR.h>
#include <cppflowCM/InterfaceCM.h>
#include <cppflowATR/InitParams.h>
#include <future>
#include <mutex>
#include <utils/loguru.hpp>

using namespace OD;

OD_CycleInput *NewCopyCycleInput(OD_CycleInput *, uint);
OD_CycleInput *SafeNewCopyCycleInput(OD_CycleInput *, uint);
void DeleteCycleInput(OD_CycleInput *);
static std::mutex glob_mutexOnNext; // not in use
static std::mutex glob_mutexOnPrev; // not in use

static bool logInitialized = false;

class ObjectDetectionManagerHandler : public ObjectDetectionManager
{
public:
    OD_ErrorCode InitObjectDetection(OD_InitParams *input);
    OD_ErrorCode StartConfigAndLogger(OD_InitParams *ip);
    OD_ErrorCode OperateObjectDetection(OD_CycleOutput *CycleOutput);
    OD_ErrorCode PrepareOperateObjectDetection(OD_CycleInput *CycleInput); // run synchroniusly
    OD_ErrorCode OperateObjectDetectionOnTiledSample(OD_CycleInput *cycleInput, OD_CycleOutput *cycleOutput);
    int PopulateCycleOutput(OD_CycleOutput *cycleOutput);
    bool SaveResultsATRimage(OD_CycleOutput *co, char *imgName, bool show);
    OD_InitParams *getParams();
    void setParams(OD_InitParams *ip);
    bool IsBusy();
    void IdleRun();
    void SetConfigParams(InitParams* ip);
    InitParams* GetConfigParams();

    std::future<OD_ErrorCode> m_result;

    ObjectDetectionManagerHandler();
    ObjectDetectionManagerHandler(OD_InitParams *ip);
    //bool WaitUntilForThread(int sec); //not in use
    bool WaitForThread();

    ~ObjectDetectionManagerHandler();

    mbInterfaceATR *m_mbATR = nullptr;
    static mbInterfaceCM *m_mbCM;

    int ApplyNMS(OD_CycleOutput *co);
    int ApplySizeMatch(OD_CycleOutput *co);


protected:
    bool InitCM();
    void DeleteAllInnerCycleInputs();
    void AnalyzeTiledSample(OD_CycleOutput *co1, std::list<float *> *tarList, OD_CycleOutput *co2);
    int CleanWrongTileDetections(OD_CycleOutput *co1, std::list<float *> *tarList);
    bool InitConfigParamsFromFile(const char *iniFilePath);
    bool InitializeLogger();
    std::string DefineATRModel(std::string nickname );
    


    OD_CycleInput *m_prevCycleInput = nullptr;
    OD_CycleInput *m_curCycleInput = nullptr;
    OD_CycleInput *m_nextCycleInput = nullptr;
    uint m_numImgPixels = 0;
    uint m_numPtrPixels = 0;
    bool m_withActiveCM = true; 
    int m_modelIndexInConfig = -1; //replaced after initialization of model 
    float m_ATR_resize_factor = -1; // will read from config, if -1 => no info 

    cv::Mat m_bigImg; 

    std::mutex m_mutexOnNext;
    std::mutex m_mutexOnPrev;
    
    static InitParams *m_configParams;
    std::string m_lastPathATR = "";

    bool m_nms = true; // use NMS as post-processing?
    bool m_size_filter = false; // use size-filter as post-processing?
    int m_nms_abs_thresh = 100; // min manhatten distance between 2 for sure distinc objects
    float m_nms_IoU_thresh = 0.3; // max IoU between 2 for sure distinc objects
    float m_nms_IoU_thresh_VEHICLE2VEHICLE = 0.3; // max IoU between 2 for sure distinc objects
    float m_nms_IoU_thresh_VEHICLE2HUMAN = 0.7; // max IoU between 2 for sure distinc objects
    float m_nms_IoU_thresh_HUMAN2HUMAN = 0.5; // max IoU between 2 for sure distinc objects
    float m_nms_IoU_thresh_VEHICLE2VEHICLE_SAME_SUB = 0.3; // max IoU between 2 for sure distinc objects
    
};
