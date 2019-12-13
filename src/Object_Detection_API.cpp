#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <cppflowATR/InterfaceATR.h>

namespace OD
{

DECLARE_API_FUNCTION ObjectDetectionManager* CreateObjectDetector(OD_InitParams * initParams)
{
    //TODO: use initParams
    ObjectDetectionManager* new_manager = new ObjectDetectionManager();


    return new_manager;

}


DECLARE_API_FUNCTION OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager* odm){
    delete odm;
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode InitObjectDetection(ObjectDetectionManager* odm, OD_InitParams * odInitParams)
{
    //TODO: initialization 
    mbInterfaceATR* mbATR = new mbInterfaceATR();
    mbATR->LoadNewModel("/home/borisef/projects/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb");
    //odm->SetMB_ATR(mbATR); //TODO
    odm->m_mbATR = mbATR;
    return OD_ErrorCode::OD_OK;

}

DECLARE_API_FUNCTION OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager* odm, OD_CycleInput* odIn, OD_CycleOutput* odOut)
{
    //TODO
    //mbATR->RunRGBimage(odIn);  
     
    //odOut = mbATR-> GetResults()
    return OD_ErrorCode::OD_OK; 
}

DECLARE_API_FUNCTION OD_ErrorCode ResetObjectDetection(ObjectDetectionManager* odm){
    return OD_ErrorCode::OD_OK; 
}

DECLARE_API_FUNCTION OD_ErrorCode GetMetry(ObjectDetectionManager* odm, int size, void *metry){
    return OD_ErrorCode::OD_OK; 
}

}