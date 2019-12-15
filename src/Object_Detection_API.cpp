#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <cppflowATR/InterfaceATR.h>

namespace OD
{

DECLARE_API_FUNCTION ObjectDetectionManager* CreateObjectDetector(OD_InitParams * initParams)
{
    // use initParams
    ObjectDetectionManager* new_manager = new ObjectDetectionManager(initParams);


    return new_manager;

}


DECLARE_API_FUNCTION OD_ErrorCode TerminateObjectDetection(ObjectDetectionManager* odm){
    delete odm;
    return OD_ErrorCode::OD_OK;
}

DECLARE_API_FUNCTION OD_ErrorCode InitObjectDetection(ObjectDetectionManager* odm, OD_InitParams * odInitParams)
{
    //TODO: initialization
    if(mbATR == NULL) 
        mbInterfaceATR* mbATR = new mbInterfaceATR();
        mbATR->LoadNewModel("/home/borisef/projects/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb");
    
    //odm->SetMB_ATR(mbATR); //TODO
    odm->setParams(odInitParams);
    odm->m_mbATR = mbATR;
    return OD_ErrorCode::OD_OK;

}

DECLARE_API_FUNCTION OD_ErrorCode OperateObjectDetectionAPI(ObjectDetectionManager* odm, OD_CycleInput* odIn, OD_CycleOutput* odOut)
{
     //TODO: keep OD_CycleInput copy

    unsigned int fi = odIn-> ImgID_input;
    int h = odm->getParams()->supportData.imageHeight;
    int w = odm->getParams()->supportData.imageWidth;
    
    e_OD_ColorImageType colortype = odm->getParams()->supportData.colorType;

    if(colortype == e_OD_ColorImageType::YUV422)// if raw
        odm->m_mbATR->RunRawImage(odIn->ptr,h,w);
    else
        if(colortype == e_OD_ColorImageType::RGB) // if rgb
            odm->m_mbATR->RunRGBVector(odIn->ptr,h,w);
        else
        {
            return OD_ErrorCode::OD_ILEGAL_INPUT;
        }
    
    // save results
    odm->PopulateCycleOutput(odOut);
    odOut->ImgID_output = fi;
    return OD_ErrorCode::OD_OK; 
}

bool  DECLARE_API_FUNCTION ObjectDetectionManager::SaveResultsATRimage(OD_CycleInput* ci,OD_CycleOutput* co, char* imgNam, bool show)
{
 //TODO:
    unsigned int fi = odIn-> ImgID_input;
    int h = odm->getParams()->supportData.imageHeight;
    int w = odm->getParams()->supportData.imageWidth;
    
    e_OD_ColorImageType colortype = odm->getParams()->supportData.colorType;
     cv::Mat* myRGB = new cv::Mat(height, width,CV_8UC1);
     unsigned char* buffer = (unsigned char*)(odIn->ptr);
     std::vector<uint8_t > img_data(h*w*3);
    if(colortype == e_OD_ColorImageType::YUV422)// if raw
        for (int i =0;i<h*w*3;i++)
            img_data[i]=buffer[i];
        convertYUV420toRGB(img_data, h, w, myRGB);
    else
        if(colortype == e_OD_ColorImageType::RGB) // if rgb
            //TODO: convertRGBbuffertoMat
        else
        {
            return false;
        }

    std::cout << "***** num_detections " << co->numOfObjects << std::endl;    
    for (int i=0; i<co->numOfObjects; i++) {
        int classId = co->ObjectsArr[i].tarClass;
        float score = co->ObjectsArr[i].tarScore;
        OD_BoundingBox bbox_data = co->ObjectsArr[i].tarBoundingBox;;
        std::vector<float> bbox = {bbox_data.x1,bbox_data.x2, bbox_data.y1, bbox_data.y2};
       
        if (score > 0.1) {
            float x = bbox[1] * w;
            float y = bbox[0] * h;
            float right = bbox[3] * w;
            float bottom = bbox[2] * h;

            cv::rectangle(*myRGB, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
        }
    }
	
    if(show){
        cv::Mat imgS;
        cv::resize(img, imgS, cv::Size(1365, 720)) ;
        cv::imshow("Image", imgS);
        cv::waitKey(0);
        }
    cv::Mat bgr(h, w,CV_8UC1);
    cv::cvtColor(*myRGB, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(imgNam, bgr);




}

int DECLARE_API_FUNCTION ObjectDetectionManager::PopulateCycleOutput(OD_CycleOutput* cycleOutput)
{
    
    OD_DetectionItem* odi = cycleOutput->ObjectsArr;

    cycleOutput->numOfObjects = m_mbATR->GetResultNumDetections();
    

    auto bbox_data = m_mbATR->GetResultBoxes();
    for (int i = 0; i < cycleOutput->numOfObjects; i++)
    {
        e_OD_TargetClass aa = e_OD_TargetClass(1);
        odi[i].tarClass = e_OD_TargetClass(m_mbATR->GetResultClasses(i));
        odi[i].tarScore = m_mbATR->GetResultScores(i);
       
        odi[i].tarBoundingBox  = {bbox_data[i * 4], bbox_data[i * 4 + 1], bbox_data[i * 4 + 2], bbox_data[i * 4 + 3]};
    }

    return 0;
}



DECLARE_API_FUNCTION OD_ErrorCode ResetObjectDetection(ObjectDetectionManager* odm){
    return OD_ErrorCode::OD_OK; 
}

DECLARE_API_FUNCTION OD_ErrorCode GetMetry(ObjectDetectionManager* odm, int size, void *metry){
    return OD_ErrorCode::OD_OK; 
}

}