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

//unsigned char *ParseImage(String path);
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
    string iniFile = (char *)"config/configATR_Feb2020_win.json";
#else
    string iniFile = (char *)"config/configATR_Feb2020.json";
#endif

    bool toDeleteATRM = true;
    bool doNotInit = false;
    int startFrameID = 1;
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
    ((ObjectDetectionManagerHandler *)atrManager)->WaitForThread();
    if (ors.toDeleteATRM)
    {
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

    OneRunStruct ors4;
    ors4.W = 4056;
    ors4.H = 3040;
    ors4.splicePath = "media/raw/*";
    ors4.imType = e_OD_ColorImageType::YUV422;
    ors4.numRepetiotions = 1;
    ors4.minDelay = 50;
    ors4.startFrameID = 100000;
    ors4.toShow = true;
    atrManager = OneRun(atrManager, ors4); 
    
    // OneRunStruct ors4nv;
    // ors4nv.W = 4056;
    // ors4nv.H = 3040;
    // ors4nv.splicePath = "media/NV12/*";
    // ors4nv.imType = e_OD_ColorImageType::NV12;
    // ors4nv.numRepetiotions = 1;
    // ors4nv.minDelay = 0;
    // ors4nv.startFrameID = 100000;
    // ors4nv.toShow = true;
    // atrManager = OneRun(atrManager, ors4nv); 


    OneRunStruct ors1;
    // ors1.H = 108000;
    // ors1.W = 192000;
    ors1.splicePath = "media/spliced/*";
    ors1.numRepetiotions = 1;
    ors1.minDelay = 0;
    ors1.startFrameID = 1;
    atrManager = OneRun(atrManager, ors1);



    OneRunStruct ors2;
    // ors2.H = 1071;
    // ors2.W = 1904;
    ors2.splicePath = "media/filter/*";
    ors2.numRepetiotions = 1;
    ors2.minDelay = 0;
    ors2.startFrameID = 1000;
    atrManager = OneRun(atrManager, ors2);

    OneRunStruct ors3;
    // ors3.W = 239300;
    // ors3.H = 186700;
    ors3.splicePath = "media/filterUCLA/*";
    ors3.numRepetiotions = 1;
    ors3.minDelay = 0;
    ors3.startFrameID = 100000;
    ors3.toDeleteATRM = false;
    atrManager = OneRun(atrManager, ors3);

    OD::ObjectDetectionManager *atrManagerAnother = nullptr;
    OneRunStruct ors5;
    // ors5.H = 1080;
    // ors5.W = 1920;
    ors5.splicePath = "media/spliced/*";
    ors5.numRepetiotions = 1;
    ors5.minDelay = 10;
    ors5.startFrameID = 500000;
    ors5.toDeleteATRM = false;

    cout<<"STRESS TEST: will create 2 at the same time"<<endl;
    atrManagerAnother = OneRun(atrManagerAnother, ors5); // will create 2 at the same time

    ors3.doNotInit = true;
    ors5.doNotInit = true;

    atrManager = OneRun(atrManager, ors3);
    atrManagerAnother = OneRun(atrManagerAnother, ors5);
    atrManager = OneRun(atrManager, ors3);
    atrManagerAnother = OneRun(atrManagerAnother, ors5);

    //cout<<"STRESS TEST: will create 3 at the same time"<<endl;
    // will create 3 at the same time // failed stressTest in PC_linux
    // OD::ObjectDetectionManager *atrManagerAnother2 = nullptr;
    // ors4.toDeleteATRM = false;
    // atrManagerAnother2 = OneRun(atrManagerAnother2, ors4); // will create 3 at the same time
    // ors4.doNotInit = true;
    // atrManager = OneRun(atrManager, ors3);
    // atrManagerAnother = OneRun(atrManagerAnother, ors5);
    // atrManagerAnother2 = OneRun(atrManagerAnother2, ors4);
    
    
    //OD::TerminateObjectDetection(atrManagerAnother2);
    OD::TerminateObjectDetection(atrManager);
    OD::TerminateObjectDetection(atrManagerAnother);
  
    cout << "Ended StressTest Normally" << endl;
    return 0;
}