#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <thread>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>
#include <cppflowATRInterface/Object_Detection_Handler.h>

using namespace std;
using namespace std::chrono;
using namespace OD;

void MyWait(string s, float ms)
{
    #ifdef TEST_MODE
    std::cout << s << " Waiting sec:" << (ms / 1000.0) << endl;
    #endif //#ifdef TEST_MODE
    std::this_thread::sleep_for(std::chrono::milliseconds((uint)ms));
}


struct OneRunStruct
{
    int H;
    int W;
    string video_path;
    int numRepetitions = 10;
    float minDelay = 0;
    bool toShow = true;
    e_OD_ColorImageType imType = e_OD_ColorImageType::RGB;


  #ifdef WIN32
    string iniFile = (char *)"config/configATR_Feb2020_win.json";
#else
    string iniFile = (char *)"config/configATR_Feb2020.json";
#endif

    bool toDeleteATRM = true;
    bool doNotInit = false;
    int startFrameID = 1;
};

OD::ObjectDetectionManager * OneRun(OD::ObjectDetectionManager *atrManager, OneRunStruct ors)
{
    // Mission
    MB_Mission mission = {
        MB_MissionType::MATMON,       //mission1.missionType
        e_OD_TargetSubClass::PRIVATE, //mission1.targetClas
        e_OD_TargetColor::WHITE       //mission1.targetColor
    };

    // support data
    OD_SupportData supportData = {
        ors.H, ors.W, //imageHeight//imageWidth
        ors.imType,   //colorType;
        100,          //rangeInMeters
        70.0f,        //fcameraAngle; //BE
        0,            //TEMP:cameraParams[10];//BE
        0             //TEMP: float	spare[3];
    };

    OD_InitParams initParams =
        {
            (char *)ors.iniFile.c_str(),
            350, // max number of items to be returned
            supportData,
            mission};

    if (ors.doNotInit == false)
    {
        // // Creation of ATR manager + new mission
        if (!atrManager)
            atrManager = OD::CreateObjectDetector(&initParams); //first mission

        cout << " ***  ObjectDetectionManager created  *** " << endl;

        // // new mission
        OD::InitObjectDetection(atrManager, &initParams);
          ((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();
    }

    //((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();

    OD_CycleInput *ci = new OD_CycleInput();
    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    OD_ErrorCode statusCycle;
    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

    cv::Mat frame;
    cv::VideoCapture cap(ors.video_path);
    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        exit;
    }

    int N = int(cap.get(cv::CAP_PROP_FRAME_COUNT));


    unsigned char *ptrTif;

    int lastReadyFrame = 0;
    co->ImgID_output = 0;
    int temp = 0;
    for (size_t i1 = 0; i1 < ors.numRepetitions; i1++)
    {
        for (size_t i = 0; i < N; i++)
        {
            temp++;

            //read next frame
            frame.release(); 
            cap >> frame;

            if(frame.empty()){
                break;
            }

            ci->ImgID_input = temp + ors.startFrameID;
            ptrTif = ParseCvMat(frame);

            ci->ptr = ptrTif;
            statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);
            if (lastReadyFrame != co->ImgID_output)
            { //draw
                cout << " Detected new results for frame " << co->ImgID_output << endl;
                string outName = "outRes/out_res3_" + std::to_string(co->ImgID_output) + ".png";
                lastReadyFrame = co->ImgID_output;
                atrManager->SaveResultsATRimage(co, (char *)outName.c_str(), ors.toShow);
            }
            MyWait("Small pause", ors.minDelay);
            delete ptrTif;
        }
    }
    //at the end
     ((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();
    if(ors.toDeleteATRM){
        OD::TerminateObjectDetection(atrManager);
        atrManager = nullptr;
    }
    //release OD_CycleInput
    delete ci;

    //release OD_CycleOutput
    delete co->ObjectsArr;
    delete co;

    return atrManager;
}

int main()
{
    OD::ObjectDetectionManager *atrManager = nullptr;

    OneRunStruct ors1;
    ors1.H = 2160;
    ors1.W = 4096;
    ors1.video_path = "media/00000018.MP4";
    ors1.numRepetitions = 1;
    ors1.minDelay = 0;
    ors1.startFrameID = 1;
    atrManager = OneRun(atrManager, ors1);


    atrManager = OneRun(atrManager, ors1);


    OD::TerminateObjectDetection(atrManager);

    cout<<"Ended VideoTest Normally"<<endl;
    return 0;
}