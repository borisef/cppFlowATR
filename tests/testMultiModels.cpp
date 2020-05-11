//
// 1) Run on thousands of images with different delays
// 2) Create and destroy managers
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
#include <cppflowATRInterface/Object_Detection_Handler.h>

using namespace std;
using namespace std::chrono;
using namespace OD;

enum enumRange
{ 
	ALL = 1,
	NEAR = 2,
	FAR = 3
};

enum enumTarget
{ 
	ANY = 1,
	CARS = 2,
	HUMANS = 3
};


vector<String> GetFileNames();

void MyWait(string s, float ms)
{
#ifdef TEST_MODE
    std::cout << s << " Waiting sec:" << (ms / 1000.0) << endl;
#endif//#ifdef TEST_MODE
    std::this_thread::sleep_for(std::chrono::milliseconds((uint)ms));
}

vector<String> GetFileNames(const char *pa)
{

    vector<String> fn;
    cv::glob(pa, fn, true);

    return fn;
}


struct OneRunStruct
{
    int H;
    int W;
    string splicePath;
    int numRepetiotions = 2;
    float minDelay = 0;
    bool toShow = true;
    e_OD_ColorImageType imType = e_OD_ColorImageType::RGB;

#ifdef WIN32
    string iniFile = (char *)"config/configATR_May2020_win.json";
#elif OS_LINUX
    #ifdef JETSON
        string iniFile = (char *)"config/configATR_May2020_linux_jetson.json";
    #else
        string iniFile = (char *)"config/configATR_May2020_linux.json";
    #endif
#endif

    bool toDeleteATRM = true; // delete at the end
    bool doNotInit = false;
    int startFrameID = 1;

    enumRange enum_range = enumRange::ALL;
    enumTarget enum_target = enumTarget::ANY;
    


};

OD::ObjectDetectionManager *OneRun(OD::ObjectDetectionManager *atrManager, OneRunStruct ors)
{
    vector<String> ff = GetFileNames((char *)ors.splicePath.c_str());
    int N = ff.size();

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

    if(ors.enum_range==NEAR)
       supportData.rangeInMeters = 100; 
    else if(ors.enum_range==FAR)
        supportData.rangeInMeters = 250;
    else
        supportData.rangeInMeters = 0;

    if(ors.enum_target==enumTarget::CARS)
       mission.targetClas = e_OD_TargetSubClass::PRIVATE; 
    else if(ors.enum_target==enumTarget::HUMANS)
        mission.targetClas = e_OD_TargetSubClass::OTHER_SUB_CLASS; 
    else //ALL
        mission.targetClas = e_OD_TargetSubClass::UNKNOWN_SUB_CLASS;
    


    // TO Avoid errors
    if (ors.imType == e_OD_ColorImageType::RGB)
    {
        cv::Mat tempim = cv::imread(ff[0]);
        supportData.imageHeight = tempim.rows;
        supportData.imageWidth = tempim.cols;
    }

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
#ifdef TEST_MODE
        cout << " ***  ObjectDetectionManager created  *** " << endl;
#endif//#ifdef TEST_MODE

        //  new mission
        OD::InitObjectDetection(atrManager, &initParams);
        ((ObjectDetectionManagerHandler *)atrManager)->WaitForThread();
    }

    //((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();

    OD_CycleInput *ci = new OD_CycleInput();
    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    OD_ErrorCode statusCycle;
    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

    unsigned char *ptrTif;
    // char *ptrTifnew;
    int lastReadyFrame = 0;
    co->ImgID_output = 0;
    int temp = 0;
    for (size_t i1 = 0; i1 < ors.numRepetiotions; i1++)
    {
        for (size_t i = 0; i < N; i++)
        {
            temp++;
            ci->ImgID_input = temp + ors.startFrameID;
            if (ors.imType == e_OD_ColorImageType::RGB)
                ptrTif = ParseImage(ff[i]);
            else
            {
                //ptrTif = ParseRaw(ff[i]);
                ptrTif = (unsigned char*)fastParseRaw(ff[i]);
                
            }

            ci->ptr = ptrTif;
            if(i % 10 != 0)
                    ci->ptr = nullptr;
            statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);
            if (lastReadyFrame != co->ImgID_output)
            { //draw
                if(i % 10 != 0)
                    cout << " *** This input is nullptr ***" << "frame "<< i <<  endl;

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
    if (ors.toDeleteATRM)
    {
        OD::TerminateObjectDetection(atrManager);
        atrManager = nullptr;
    }
    else
         ((ObjectDetectionManagerHandler *)atrManager)->WaitForThread();
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
    ors1.splicePath = "media/spliced/*";
    ors1.numRepetiotions = 1;
    ors1.minDelay = 10;
    ors1.startFrameID = 1;
    
    ors1.toDeleteATRM = false;

    ors1.startFrameID += 1000;
    ors1.enum_range = enumRange::ALL;
    ors1.enum_target = enumTarget::ANY;

    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

  
    ors1.enum_range = enumRange::NEAR;
    ors1.enum_target = enumTarget::ANY;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

    ors1.enum_range = enumRange::FAR;
    ors1.enum_target = enumTarget::ANY;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

    ors1.startFrameID += 1000;
    ors1.enum_range = enumRange::ALL;
    ors1.enum_target = enumTarget::CARS;

    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

  
    ors1.enum_range = enumRange::NEAR;
    ors1.enum_target= enumTarget::CARS;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

    ors1.enum_range = enumRange::FAR;
    ors1.enum_target= enumTarget::CARS;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

    ors1.startFrameID += 1000;
    ors1.enum_range = enumRange::ALL;
    ors1.enum_target= enumTarget::HUMANS;

    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

  
    ors1.enum_range = enumRange::NEAR;
    ors1.enum_target= enumTarget::HUMANS;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;

    ors1.enum_range = enumRange::FAR;
    ors1.enum_target= enumTarget::HUMANS;
    ors1.startFrameID += 1000;
    
    atrManager = OneRun(atrManager, ors1);
    OD::TerminateObjectDetection(atrManager); atrManager = nullptr;


    cout << "Ended StressTest Normally" << endl;
    return 0;
}