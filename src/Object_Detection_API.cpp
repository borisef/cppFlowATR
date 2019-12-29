#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <cppflowATRInterface/Object_Detection_Handler.h>
#include <cppflowATR/InterfaceATR.h>
#include <utils/imgUtils.h>

namespace OD
{
DECLARE_API_FUNCTION ObjectDetectionManager::ObjectDetectionManager()
{
    m_initParams = nullptr;
}
DECLARE_API_FUNCTION ObjectDetectionManager::ObjectDetectionManager(OD_InitParams *ip)
{
    m_initParams = ip;
}

DECLARE_API_FUNCTION ObjectDetectionManager *CreateObjectDetector(OD_InitParams *initParams)
{
    // use initParams
    cout << "Creating graph from" << initParams->iniFilePath << std::endl;

    ObjectDetectionManager *new_manager;
    ObjectDetectionManagerHandler *new_managerH = new ObjectDetectionManagerHandler(initParams);
    new_manager = (ObjectDetectionManager *)new_managerH;
    cout << "Created ObjectDetectionManager " << endl;

    return new_manager;
}

DECLARE_API_FUNCTION OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager *odm)
{
    if (odm != nullptr)
    {
        delete (ObjectDetectionManagerHandler *)odm; //delete as handler
        odm = nullptr;
    }
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode InitObjectDetection(ObjectDetectionManager *odm, OD_InitParams *odInitParams)
{
    cout << "Entering InitObjectDetection" << endl;

    ObjectDetectionManagerHandler *odmHandler = (ObjectDetectionManagerHandler *)odm;

    OD_ErrorCode ec = odmHandler->InitObjectDetection(odInitParams);

    cout << "Finished InitObjectDetection" << endl;

    return ec;
}

DECLARE_API_FUNCTION OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager *odm, OD_CycleInput *odIn, OD_CycleOutput *odOut)
{
    OD_ErrorCode ec = OD_ErrorCode::OD_OK;
    ObjectDetectionManagerHandler *odmHandler = (ObjectDetectionManagerHandler *)odm;

    OD_ErrorCode prepOD = odmHandler->PrepareOperateObjectDetection(odIn, odOut); // run synchroniusly

    if (prepOD == OD_ErrorCode::OD_OK && !odmHandler->IsBusy())
    {
        cout << "+++Can  Operate OD... Free for step " << odIn->ImgID_input << endl;
        //ec = odmHandler->OperateObjectDetection(odIn, odOut); // synchroniously
        odmHandler->m_result = std::async(std::launch::async, &ObjectDetectionManagerHandler::OperateObjectDetection, odmHandler, odIn, odOut);
    }
    else
    {
        cout << "---Can not Operate OD... Busy for step " << odIn->ImgID_input << endl;
    }

    return ec; // TODO: not always ok
}

bool DECLARE_API_FUNCTION ObjectDetectionManager::SaveResultsATRimage(OD_CycleInput *ci, OD_CycleOutput *co, char *imgNam, bool show)
{
    return ((ObjectDetectionManagerHandler *)this)->SaveResultsATRimage(ci, co, imgNam, show);
}

DECLARE_API_FUNCTION OD_ErrorCode ResetObjectDetection(ObjectDetectionManager *odm)
{
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode GetMetry(ObjectDetectionManager *odm, int size, void *metry)
{
    return OD_ErrorCode::OD_OK;
}

} // namespace OD