#include <cppflowATRInterface/Object_Detection_Handler.h>
#include <utils/imgUtils.h>

#include <iomanip>
#include <future>

using namespace OD;
using namespace std;



OD_InitParams* ObjectDetectionManagerHandler::getParams(){return m_initParams;}

void ObjectDetectionManagerHandler::setParams(OD_InitParams* ip)
{
    m_initParams = ip;
    // if(m_cycleInput.ptr)
    // {
    //         delete m_cycleInput.ptr;
    //         m_cycleInput.ptr = new unsigned char[(ip->supportData.imageWidth)*(ip->supportData.imageHeight)*3];
    // }
}


ObjectDetectionManagerHandler::ObjectDetectionManagerHandler()
{
    m_mbATR = nullptr;
    //m_cycleInput.ptr = nullptr;

}
ObjectDetectionManagerHandler::ObjectDetectionManagerHandler(OD_InitParams*   ip):ObjectDetectionManagerHandler()
{
    setParams(ip);
}

ObjectDetectionManagerHandler::~ObjectDetectionManagerHandler()
{

      if (m_mbATR != nullptr)
        delete m_mbATR;
}

OD_ErrorCode ObjectDetectionManagerHandler::InitObjectDetection(OD_InitParams* odInitParams) 
{
    mbInterfaceATR *mbATR = nullptr;
     //initialization
     if (m_mbATR == nullptr)
    {
        mbATR = new mbInterfaceATR();
        cout<<"Create new mbInterfaceATR in ObjectDetectionManagerHandler::InitObjectDetection"<<endl;
        //TODO: decide which model to take (if to take or stay with old )
        mbATR->LoadNewModel(odInitParams->iniFilePath);
        m_mbATR = mbATR;
        cout<<"Executed LoadNewModel in  InitObjectDetection"<<endl;
    }

    setParams(odInitParams);
   


    return OD_ErrorCode::OD_OK;
}
OD_ErrorCode ObjectDetectionManagerHandler::PrepareOperateObjectDetection(OD_CycleInput* CycleInput, OD_CycleOutput* CycleOutput)
{
    //keep OD_CycleInput copy
    cout<<" PrepareOperateObjectDetection: Prepare to run on frame "<< CycleInput->ImgID_input<<endl;
    //int wh3=(m_initParams->supportData.imageHeight)*(m_initParams->supportData.imageWidth)*3;
   // std::copy((CycleInput->ptr),(CycleInput->ptr) + wh3, (char*)m_cycleInput.ptr);
   // m_cycleInput.ImgID_input = CycleInput->ImgID_input;

    //remark: can be skipped for performance but may be risky 
    //TODO: during prepare we can convert to RGB

    return OD_ErrorCode::OD_OK;

}
OD_ErrorCode  ObjectDetectionManagerHandler::OperateObjectDetection(OD_CycleInput* odIn, OD_CycleOutput* odOut) 
{
    m_isBusy = true; //LOCK
    cout<<"^^^Locked"<<endl;
    cout<<" OperateObjectDetection: Run on frame "<< odIn->ImgID_input<<endl;

    unsigned int fi = odIn->ImgID_input;
    int h = m_initParams->supportData.imageHeight;
    int w = m_initParams->supportData.imageWidth;

    e_OD_ColorImageType colortype = m_initParams->supportData.colorType;

    if (colortype == e_OD_ColorImageType::YUV422) // if raw
        this->m_mbATR->RunRawImage(odIn->ptr, h, w);
    else if (colortype == e_OD_ColorImageType::RGB) // if rgb
    {
        cout << " Internal Run on RGB buffer " << endl;
        this->m_mbATR->RunRGBVector(odIn->ptr, h, w);
    }
    else
    {
        return OD_ErrorCode::OD_ILEGAL_INPUT;
    }

    // save results
    this->PopulateCycleOutput(odOut);
    odOut->ImgID_output = fi;

    
    cout<<"###UnLocked"<<endl;
    m_isBusy = false;//RELEASE

    return OD_ErrorCode::OD_OK;
}

bool  ObjectDetectionManagerHandler::SaveResultsATRimage(OD_CycleInput *ci, OD_CycleOutput *co, char *imgNam, bool show)
{
    //TODO:
    unsigned int fi = ci->ImgID_input;
    unsigned int h = m_initParams->supportData.imageHeight;
    unsigned int w = m_initParams->supportData.imageWidth;

    e_OD_ColorImageType colortype = m_initParams->supportData.colorType;
    cv::Mat *myRGB = nullptr;
    unsigned char *buffer = (unsigned char *)(ci->ptr);
    std::vector<uint8_t> img_data(h * w * 2);
    if (colortype == e_OD_ColorImageType::YUV422) // if raw
    {

        for (int i = 0; i < h * w * 2; i++)//TODO: without for loop
            img_data[i] = buffer[i];

        myRGB = new cv::Mat(h, w, CV_8UC3);
        convertYUV420toRGB(img_data, w, h, myRGB);
    }
    else if (colortype == e_OD_ColorImageType::RGB) // if rgb
    {
        myRGB = new cv::Mat(h, w, CV_8UC3);
        //myRGB->data = buffer;// NOT safe if we going to use buffer later
        std::copy(buffer,buffer+w*h*3,myRGB->data);
        //std::copy(arrayOne, arrayOne+10, arrayTwo);
        cv::imwrite("debug_newImg1_bgr.tif", *myRGB);
    }
    else
    {
        return false;
    }

    std::cout << "***** num_detections " << co->numOfObjects << std::endl;
    for (int i = 0; i < co->numOfObjects; i++)
    {
        int classId = co->ObjectsArr[i].tarClass;
        float score = co->ObjectsArr[i].tarScore;
        OD_BoundingBox bbox_data = co->ObjectsArr[i].tarBoundingBox;

        std::vector<float> bbox = {bbox_data.x1, bbox_data.x2, bbox_data.y1, bbox_data.y2};

        if (score > 0.1)
        {
            cout << "add rectangle to drawing" <<endl;
            float x = bbox_data.x1;
            float y = bbox_data.y1;
            float right = bbox_data.x2;
            float bottom = bbox_data.y2;

            cv::rectangle(*myRGB, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
            cv::putText(*myRGB,string("Label:") + std::to_string(classId) + ";" + std::to_string(int(score*100)) + "%",cv::Point(x,y - 10),1, 2, Scalar(124,200,10),3);
            //cv::putText(*myRGB, string("Label:") + std::to_string(classId) , cv::Point(x,y + 5),1, 2, Scalar(124,100,0),3);
        }
    }
    cout << " Done reading targets" << endl;
    if (show)
    {
        cv::Mat imgS;
        cv::resize(*myRGB, imgS, cv::Size(1365, 720));
        cv::imshow("Image", imgS);
       // cv::waitKey(0);
    }
    cv::Mat bgr(h, w, CV_8UC3);
    cv::cvtColor(*myRGB, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(imgNam, bgr);
     cout << " Done saving image" << endl;
    if(myRGB != nullptr){
        myRGB->release();
        delete myRGB; // TODO
        
    }
    cout << " Done cleaning image" << endl;
    return true;

}

int  ObjectDetectionManagerHandler::PopulateCycleOutput(OD_CycleOutput *cycleOutput)
{
    
    cout<<"ObjectDetectionManagerHandler::PopulateCycleOutput"<<endl;

    OD_DetectionItem *odi = cycleOutput->ObjectsArr;

    cycleOutput->numOfObjects = m_mbATR->GetResultNumDetections();

    auto bbox_data = m_mbATR->GetResultBoxes();
    unsigned int w = this->m_initParams->supportData.imageWidth;
    unsigned int h = this->m_initParams->supportData.imageHeight;
    for (int i = 0; i < cycleOutput->numOfObjects; i++)
    {
        e_OD_TargetClass aa = e_OD_TargetClass(1);
        odi[i].tarClass = e_OD_TargetClass(m_mbATR->GetResultClasses(i));
        odi[i].tarScore = m_mbATR->GetResultScores(i);
        
       // odi[i].tarBoundingBox = {bbox_data[i * 4], bbox_data[i * 4 + 1], bbox_data[i * 4 + 2], bbox_data[i * 4 + 3]};
        odi[i].tarBoundingBox = {bbox_data[i * 4 + 1]*w, bbox_data[i * 4 + 3]*w, bbox_data[i * 4]*h,bbox_data[i * 4 + 2]*h};

    }

    return cycleOutput->numOfObjects;
}