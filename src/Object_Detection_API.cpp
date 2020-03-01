#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <cppflowATRInterface/Object_Detection_Handler.h>
#include <cppflowATR/InterfaceATR.h>
#include <utils/imgUtils.h>

namespace OD
{
#ifndef WIN32
DECLARE_API_FUNCTION ObjectDetectionManager::ObjectDetectionManager()
{
    m_initParams = nullptr;
}
DECLARE_API_FUNCTION ObjectDetectionManager::ObjectDetectionManager(OD_InitParams *ip)
{
    m_initParams = ip;
}
#else

ObjectDetectionManager::ObjectDetectionManager()
{
    m_initParams = nullptr;
}

ObjectDetectionManager::ObjectDetectionManager(OD_InitParams *ip)
{
    m_initParams = ip;
}

#endif
ObjectDetectionManager *CreateObjectDetector(OD_InitParams *initParams)
{
    #ifdef TEST_MODE
    cout << "Creating ObjectDetectionManager from ini" << initParams->iniFilePath << std::endl;
    #endif //TEST_MODE
    

    ObjectDetectionManager *new_manager;
    ObjectDetectionManagerHandler *new_managerH = new ObjectDetectionManagerHandler(initParams);
    new_manager = (ObjectDetectionManager *)new_managerH;

    #ifdef TEST_MODE
    cout << "Created ObjectDetectionManager " << endl;
    #endif //TEST_MODE

    return new_manager;
}

DECLARE_API_FUNCTION OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager *);
OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager *odm)
{
    if (odm != nullptr)
    {
        ((ObjectDetectionManagerHandler *)odm)->WaitForThread();
        delete (ObjectDetectionManagerHandler *)odm; //delete as handler
        odm = nullptr;
    }
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode InitObjectDetection(ObjectDetectionManager *, OD_InitParams *);
OD_ErrorCode InitObjectDetection(ObjectDetectionManager *odm, OD_InitParams *odInitParams)
{
    #ifdef TEST_MODE
    cout << "Entering InitObjectDetection" << endl;
    #endif //TEST_MODE

    ObjectDetectionManagerHandler *odmHandler = (ObjectDetectionManagerHandler *)odm;
    OD_ErrorCode ec0 = odmHandler->StartConfigAndLogger(odInitParams);
     //From now we can use logger !!!
     //TODO: what if ec0 is not OK ? 

    OD_ErrorCode ec = odmHandler->InitObjectDetection(odInitParams);

    #ifdef TEST_MODE
    cout << "Finished InitObjectDetection" << endl;
    #endif //TEST_MODE
    
    return ec;
}

DECLARE_API_FUNCTION OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager *, OD_CycleInput *, OD_CycleOutput *);
OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager *odm, OD_CycleInput *odIn, OD_CycleOutput *odOut)
{
    OD_ErrorCode ec = OD_ErrorCode::OD_OK;
    ObjectDetectionManagerHandler *odmHandler = (ObjectDetectionManagerHandler *)odm;

    if(odmHandler->getParams()->mbMission.missionType == OD::MB_MissionType::ANALYZE_SAMPLE)
        {
            odmHandler->OperateObjectDetectionOnTiledSample(odIn,odOut);
            return ec; // TODO: not always ok
        }



    OD_ErrorCode prepOD = odmHandler->PrepareOperateObjectDetection(odIn); // run synchroniusly

    if (prepOD == OD_ErrorCode::OD_OK && !odmHandler->IsBusy())
    {
        #ifdef TEST_MODE
        cout << "+++Can  Operate OD... Free for step " << odIn->ImgID_input << endl;
        #endif //TEST_MODE

        //ec = odmHandler->OperateObjectDetection(odOut); // synchroniously
        odmHandler->m_result = std::async(std::launch::async, &ObjectDetectionManagerHandler::OperateObjectDetection, odmHandler, odOut);
    }
    else
    {
        #ifdef TEST_MODE
        cout << "---Can not Operate OD... Busy for step " << odIn->ImgID_input << endl;
        #endif //TEST_MODE
    }

    return ec; // TODO: not always ok
}

bool ObjectDetectionManager::SaveResultsATRimage(OD_CycleOutput *co, char *imgNam, bool show)
{
    return ((ObjectDetectionManagerHandler *)this)->SaveResultsATRimage(co, imgNam, show);
}

DECLARE_API_FUNCTION OD_ErrorCode ResetObjectDetection(ObjectDetectionManager *odm);
OD_ErrorCode ResetObjectDetection(ObjectDetectionManager *odm)
{
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode GetMetry(ObjectDetectionManager *odm, int size, void *metry);
OD_ErrorCode GetMetry(ObjectDetectionManager *odm, int size, void *metry)
{
    return OD_ErrorCode::OD_OK;
}

} // namespace OD