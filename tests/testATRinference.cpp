//
// Created by BE.
// Changed be
//

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

using namespace std;
using namespace std::chrono;

int main()
{

#ifdef TEST_MODE
    cout << "Test Mode" << endl;
#endif

    bool SHOW = true;
    float numIter = 3.0;
#ifdef WIN32
    unsigned int W = 1292;
    unsigned int H = 969;
#else
    unsigned int W = 4096;
    unsigned int H = 2160;
#endif
    unsigned int frameID = 42;

    // Mission
    MB_Mission mission1 = {
        MB_MissionType::MATMON,       //mission1.missionType
        e_OD_TargetSubClass::PRIVATE, //mission1.targetClas
        e_OD_TargetColor::WHITE       //mission1.targetColor
    };

    // support data
    OD_SupportData supportData1 = {
        H, W,                     //imageHeight//imageWidth
        e_OD_ColorImageType::RGB, //colorType;
        100,                      //rangeInMeters
        70.0f,                    //fcameraAngle; //BE
        0,                        //TEMP:cameraParams[10];//BE
        0                         //TEMP: float	spare[3];
    };

    OD_InitParams initParams1 =
        {
    //(char*)"/home/magshim/MB2/TrainedModels/faster_MB_140719_persons_sel4/frozen_390k/frozen_inference_graph.pb", //fails
    //(char*)"tryTRT_humans.pb", //sometimes works
    //(char*)"/home/magshim/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb", //works
//(char*)"tryTRT_humans.pb", //sometimes OK, sometimes crashes the system
#ifdef WIN32
            (char *)"graphs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03_frozen_inference_graph.pb",
#else
            (char *)"graphs/frozen_inference_graph_humans.pb",
#endif
            //  (char*)"tryTRT_all.pb", //Nope
            // (char*)"/home/magshim/cppflowATR/frozen_inference_graph_all.pb",
            350, // max number of items to be returned
            supportData1,
            mission1};

    // Creation of ATR manager + new mission
    OD::ObjectDetectionManager *atrManager;
    atrManager = OD::CreateObjectDetector(&initParams1); //first mission

    cout << " ***  ObjectDetectionManager created  *** " << endl;

    // new mission
    OD::InitObjectDetection(atrManager, &initParams1);

    //emulate buffer from TIF
    cout << " ***  Read tif image to rgb buffer  ***  " << endl;

#ifdef WIN32
    cv::Mat inp1 = cv::imread("media/girl.jpg", CV_LOAD_IMAGE_COLOR);
#else
    cv::Mat inp1 = cv::imread("media/00000018.tif", CV_LOAD_IMAGE_COLOR);
#endif

    cv::cvtColor(inp1, inp1, CV_BGR2RGB);

    //put image in vector
    std::vector<uint8_t> img_data1;
    img_data1.assign(inp1.data, inp1.data + inp1.total() * inp1.channels());

    unsigned char *ptrTif = new unsigned char[img_data1.size()];
    std::copy(begin(img_data1), end(img_data1), ptrTif);

    OD_CycleInput *ci = new OD_CycleInput();
    // ci->ImgID_input = 42;
    ci->ptr = ptrTif;

    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    OD_ErrorCode statusCycle;

    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = 350;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

    // RUN ONE EMPTY CYCLE
    statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);
    float milliseconds = 20000;
    std::cout << "Empty Waiting sec:" << (milliseconds / 1000.0) << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds((uint)milliseconds));

    co->ImgID_output = 0;

    uint lastReadyFrame = 0;
    for (int i = 1; i < 100; i++)
    {
        ci->ImgID_input = 0 + i;
        // std::copy(begin(img_data1), end(img_data1), ptrTif);
        // ci->ptr = ptrTif;

        cout << " ***  Run inference on RGB image  ***  step " << i << endl;

        statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);

        cout << " ***  .... After inference on RGB image   *** " << endl;

        if (lastReadyFrame != co->ImgID_output)
        { //draw
            cout << " Detected new results for frame " << co->ImgID_output << endl;
            string outName = "outRes/out_res1_" + std::to_string(co->ImgID_output) + ".png";
            lastReadyFrame = co->ImgID_output;
            atrManager->SaveResultsATRimage(ci, co, (char *)outName.c_str(), false);
        }

        float milliseconds = 25;
        std::this_thread::sleep_for(std::chrono::milliseconds((uint)milliseconds));
        std::cout << "Waited sec:" << (milliseconds / 1000.0) << endl;
    }

    //release buffer
    delete ptrTif;

    W = 4056;
    H = 3040;
    frameID++;

    // change  support data
    OD_SupportData supportData2 = {
        H, W,                        //imageHeight//imageWidth
        e_OD_ColorImageType::YUV422, // colorType;
        100,                         //rangeInMeters
        70.0f,                       //fcameraAngle; //BE
        0,                           //TEMP:cameraParams[10];//BE
        0                            //TEMP: float	spare[3];
    };

    initParams1.supportData = supportData2;
    // new mission because of support data
    InitObjectDetection(atrManager, &initParams1);

    //emulate buffer from RAW
    std::vector<unsigned char> vecFromRaw = readBytesFromFile("media/00006160.raw");

    unsigned char *ptrRaw = new unsigned char[vecFromRaw.size()];
    std::copy(begin(vecFromRaw), end(vecFromRaw), ptrRaw);

    ci->ptr = ptrRaw; //use same ci and co
    lastReadyFrame = 0;
    for (int i = 0; i < 100; i++)
    {
        ci->ImgID_input = 0 + i;
        cout << " ***  Run inference on RAW image  ***  " << endl;

        statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);

        cout << " ***  .... After inference on RAW image   *** " << endl;

        if (lastReadyFrame != co->ImgID_output)
        { //draw
            cout << " Detected new results for frame " << co->ImgID_output << endl;
            string outName = "outRes/out_res2_" + std::to_string(co->ImgID_output) + ".png";
            lastReadyFrame = co->ImgID_output;
            atrManager->SaveResultsATRimage(ci, co, (char *)outName.c_str(), false);
        }

        float milliseconds = 10;
        std::this_thread::sleep_for(std::chrono::milliseconds((uint)milliseconds));
        std::cout << "Waited sec:" << (milliseconds / 1000.0) << endl;
    }
    atrManager->SaveResultsATRimage(ci, co, (char *)"out_res2.tif", false);

    delete ptrRaw;

    //at the end
    OD::TerminateObjectDetection(atrManager);

    //release OD_CycleInput
    delete ci;

    //release OD_CycleOutput
    delete co->ObjectsArr;
    delete co;

    return 0;
}
