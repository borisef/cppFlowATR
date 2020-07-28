#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include "inih/INIReader.h"

#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <thread>

#include <iostream>
#include <exception>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>

#define CONFIG_PATH "config.ini"
#define DEFAULT_INPUT String path, OD_CycleInput *ci, OD_CycleOutput *co, OD::ObjectDetectionManager *atrManager

INIReader *reader;
unsigned int last_frame;

OD_InitParams *LoadConfig();
unsigned char *ParseImage(cv::Mat inp);
void doInference(cv::Mat img, OD::ObjectDetectionManager *atrManager, OD_CycleInput *ci, OD_CycleOutput *co);
void image(String path, OD_CycleInput *ci, OD_CycleOutput *co, OD::ObjectDetectionManager *atrManager);
void video(String path, OD_CycleInput *ci, OD_CycleOutput *co, OD::ObjectDetectionManager *atrManager);

int main()
{
    //make objects
    auto config = LoadConfig();
    OD::ObjectDetectionManager *atrManager = OD::CreateObjectDetector(config);
    OD::InitObjectDetection(atrManager, config);

    OD_CycleInput *ci = new OD_CycleInput();
    OD_CycleOutput *co = new OD_CycleOutput();

    String test = "test1";

    String path = reader->GetString(test, "path", "err");
    String ext = path.substr(path.find_last_of(".")).toLowerCase();

    if (ext == ".tif" || ext == ".jpg" || ext == ".png")
    {
        //image
        image(path, ci, co, atrManager);
    }
    else if (ext == ".mp4")
    {
        //video
        video(path, ci, co, atrManager);
    }
    else if (ext == "")
    {
        //folder
        //dir(path, ci, co, atrManager);
    }
}

void MyWait(string s, float ms)
{
    std::cout << s << " Waiting sec:" << (ms / 1000.0) << endl;
    std::this_thread::sleep_for(std::chrono::milliseconds((uint)ms));
}

void image(String path, OD_CycleInput *ci, OD_CycleOutput *co, OD::ObjectDetectionManager *atrManager)
{
      #ifdef OPENCV_MAJOR_4
    cv::Mat img = imread(path, IMREAD_COLOR );//CV_LOAD_IMAGE_COLOR
    #else
    cv::Mat img = imread(path, CV_LOAD_IMAGE_COLOR );//CV_LOAD_IMAGE_COLOR
    #endif
    if (!img.data) // Check for invalid input
    {
        cout << "Could not open or find the image" << std::endl;
        throw "Could not open or find the image";
    }

    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = 350;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

    ci->ImgID_input = 0;

    doInference(img, atrManager, ci, co);
    MyWait("Long pause", 10000.0);
}

void video(String path, OD_CycleInput *ci, OD_CycleOutput *co, OD::ObjectDetectionManager *atrManager)
{
    cv::VideoCapture cap(path);

    if (!cap.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        throw "Error opening video stream or file";
    }

    cv::Mat frame;
    while (1)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        doInference(frame, atrManager, ci, co);
    }
}

void dir(DEFAULT_INPUT)
{
}

void doInference(cv::Mat img, OD::ObjectDetectionManager *atrManager, OD_CycleInput *ci, OD_CycleOutput *co)
{
    ci->ptr = ParseImage(img);

    cout << " ***  Run inference on RGB image  *** " << endl;
    OD_ErrorCode statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);
    cout << " ***  .... After inference on RGB image   *** " << endl;
    
    if (last_frame != co->ImgID_output)
    {
        cout << " Detected new results for frame " << co->ImgID_output << endl;
        string outName = "output/out" + std::to_string(co->ImgID_output) + ".png";
        last_frame = co->ImgID_output;
        atrManager->SaveResultsATRimage(ci, co, (char *)outName.c_str(), true);
    }
    
    MyWait("Small pause", 100.0);
}

unsigned char *ParseImage(cv::Mat inp)
{
    //cv::cvtColor(inp, inp, CV_BGR2RGB);

    //put image in vector
    std::vector<uint8_t> img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());

    unsigned char *ptrTif = new unsigned char[img_data.size()];
    std::copy(begin(img_data), end(img_data), ptrTif);

    return ptrTif;
}
OD_InitParams *LoadConfig()
{
    reader = new INIReader(CONFIG_PATH);

    if (reader->ParseError() < 0)
    {
        throw "error loading config file";
    }

    bool SHOW = true;
    float numIter = 3.0;

    unsigned int height = (unsigned int)reader->GetInteger("supportData", "height", 0);
    unsigned int width = (unsigned int)reader->GetInteger("supportData", "width", 0);

    MB_Mission *mission1 = new MB_Mission();
    *mission1 = {
        MB_MissionType::MATMON,       //mission1.missionType
        e_OD_TargetSubClass::PRIVATE, //mission1.targetSubClass
        e_OD_TargetColor::WHITE       //mission1.targetColor
    };

    // support data
    OD_SupportData *supportData1 = new OD_SupportData();
    *supportData1 = {
        width,
        height,
        e_OD_ColorImageType::RGB, //colorType;
        100,                      //rangeInMeters
        70.0f,                    //fcameraAngle; //BE
        0,                        //TEMP:cameraParams[10];//BE
        0                         //TEMP: float	spare[3];
    };

    string *graph = new string();
    *graph = reader->Get("initParams", "graph", "err");

    OD_InitParams *initParams = new OD_InitParams();
    *initParams = {
        graph->c_str(),                                    //graph
        reader->GetInteger("initParams", "max_items", -1), // max number of items to be returned
        *supportData1,
        *mission1};

    return initParams;
}