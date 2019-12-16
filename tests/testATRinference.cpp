//
//
// Created BE
//

//#include "cppflow/Model.h"
//#include "cppflow/Tensor.h"
//#include "cppflowATR/InterfaceATR.h"
#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>

#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;

using namespace OD;

int main()
{
    cout << " *** begin ***" << endl;
    unsigned int H = 4056;
    unsigned int W = 3040;
    unsigned int frameID = 42;

    // Mission
    MB_Mission mission1 = {
        MB_MissionType::MATMON,       //mission1.missionType
        e_OD_TargetSubClass::PRIVATE, //mission1.targetClas
        e_OD_TargetColor::WHITE       //mission1.targetColor
    };
    // support data
    OD_SupportData supportData1 = {
        H, W,                        //imageHeight//imageWidth
        e_OD_ColorImageType::YUV422, // colorType;
        100,                         //rangeInMeters
        70.0f,                       //fcameraAngle; //BE
        0,                           //TEMP:cameraParams[10];//BE
        0                            //TEMP: float	spare[3];
    };

    // ATR params
    OD_InitParams *initParams1 = new OD_InitParams();
    initParams1->iniFilePath = (char *)"inifile.txt"; // path to ini file
    initParams1->numOfObjects = 100;                  // max number of items to be returned
    initParams1->supportData = supportData1;
    initParams1->mbMission = mission1;

    // Creation of ATR manager + new mission
    ObjectDetectionManager *atrManager;
    atrManager = CreateObjectDetector(initParams1); //first mission

    cout << " ***  ObjectDetectionManager created  *** " << endl;

    // new mission
    InitObjectDetection(atrManager, initParams1);

    //emulate buffer from RAW
    std::vector<unsigned char> vecFromRaw = readBytesFromFile("/home/borisef/projects/cppflowATR/00006160.raw");

    unsigned char *ptr1 = new unsigned char[vecFromRaw.size()];
    std::copy(begin(vecFromRaw), end(vecFromRaw), ptr1);

    OD_CycleInput *ci1 = new OD_CycleInput(frameID, ptr1);
    //ci1->ptr = ptr;

    OD_CycleOutput *co = new OD_CycleOutput(initParams1->numOfObjects); // allocate empty cycle output buffer
    OD_ErrorCode statusCycle;

    for(int i=0;i<10;i++)
    {
    cout << " ***  Run inference on RAW image  ***  " << endl;

    statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci1, co);

    cout << " ***  .... After inference on RAW image   *** " << endl;
    }
    //draw
    atrManager->SaveResultsATRimage(ci1, co, (char *)"out_res1.tif", false);

    //Destroy cycle input
    delete ptr1;
    delete ci1;

    //run on image from file
    OD_SupportData supportData2 = {
        4096, 2160,               //imageHeight//imageWidth
        e_OD_ColorImageType::RGB, // colorType;
        100,                      //rangeInMeters
        70.0f,                    //fcameraAngle; //BE
        0,                        //cameraParams[10];//BE
        0                         //float	spare[3];
    };
    initParams1->supportData = supportData2;
    // new mission same initParams1
    OD::InitObjectDetection(atrManager, initParams1);

    //emulate buffer from TIF

    cout << " ***  Read tif image to rgb buffer  ***  " << endl;

    cv::Mat inp = cv::imread("00000018.tif", CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(inp, inp, CV_BGR2RGB);
    // //put image in vector
    std::vector<uint8_t> img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());

    unsigned char *ptr2 = new unsigned char[img_data.size()];
    std::copy(begin(img_data), end(img_data), ptr2);

    OD_CycleInput *ci2 = new OD_CycleInput(frameID + 1, ptr2);

    for(int i=0;i<10;i++)
    {
        cout << " ***  Call inference on RGB buffer   *** " << endl;
        statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci2, co);
        cout << "  ***  ... After inference on RGB buffer   *** " << endl;
    }
    //draw
    atrManager->SaveResultsATRimage(ci2, co, (char *)"out_res2.tif", false);

    cout << "  ***  ... After SaveResultsATRimage on RGB buffer   *** " << endl;

    //Destroy cycle input /output
    delete ptr2;
    delete ci2;
    delete co;

    cout << " **Delete stuff***" << endl;

    DeleteObjectDetection(atrManager);
    delete initParams1;
}
