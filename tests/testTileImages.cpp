#include <utils/imgUtils.h>

#include <opencv2/opencv.hpp>
//#include <e:/Installs/opencv/sources/include/opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>
#include <cppflowATRInterface/Object_Detection_Handler.h>

using namespace std;
using namespace std::chrono;
using namespace OD;

OD_CycleOutput* NewOD_CycleOutput(int maxNumOfObjects, int defaultImgID_output = 0){
    
    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    co->maxNumOfObjects = maxNumOfObjects;
    co->ImgID_output = defaultImgID_output;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];
    return co;
}

void swap_OD_DetectionItem(OD_DetectionItem* xp, OD_DetectionItem * yp)  
{  
    OD_DetectionItem temp = *xp;  
    *xp = *yp;  
    *yp = temp;  
}  
void bubbleSort_OD_DetectionItem(OD_DetectionItem* arr, int n)  
{  
    int i, j;  
    for (i = 0; i < n-1; i++)      
      
    // Last i elements are already in place  
    for (j = 0; j < n-i-1; j++)  
        if (arr[j].tarScore < arr[j+1].tarScore)  
            swap_OD_DetectionItem(&arr[j], &arr[j+1]);  
}  

void AnalyzeTiles(OD_CycleOutput *co1, std::list<float *> *tarList, OD_CycleOutput *co2)
{
    int MAX_TILES_CONSIDER = 3;

    co2->maxNumOfObjects = 100;
    co2->numOfObjects = 0;
    co2->ObjectsArr = new OD_DetectionItem[co2->maxNumOfObjects];

    //temp

    for (size_t i = 0; i < co1->numOfObjects; i++)
    {
        cout << co1->ObjectsArr[i].tarClass << endl;
        cout << co1->ObjectsArr[i].tarSubClass << endl;
        cout << co1->ObjectsArr[i].tarColor << endl;
        //TODO: make sure co1->ObjectsArr[i] is one of tarList[j]
        // if already exists increment score
        int targetSlot = co2->numOfObjects;
        for (size_t i1 = 0; i1 < co2->numOfObjects; i1++)
        {
            if (co1->ObjectsArr[i].tarClass == co2->ObjectsArr[i1].tarClass)
                if (co1->ObjectsArr[i].tarSubClass == co2->ObjectsArr[i1].tarSubClass)
                    if (co1->ObjectsArr[i].tarColor == co2->ObjectsArr[i1].tarColor)
                    {
                        targetSlot = i1;
                        break;
                    }
        }
       
        // if not add element co2->numOfObjects, co2->numOfObjects++
        if (targetSlot == co2->numOfObjects)
        {
             if(co2->maxNumOfObjects <= co2->numOfObjects)//jic
                continue;
            co2->numOfObjects = co2->numOfObjects + 1;
            co2->ObjectsArr[targetSlot].tarScore = 0;
           
        }
        co2->ObjectsArr[targetSlot].tarClass = co1->ObjectsArr[i].tarClass;
        co2->ObjectsArr[targetSlot].tarSubClass = co1->ObjectsArr[i].tarSubClass;
        co2->ObjectsArr[targetSlot].tarColor = co1->ObjectsArr[i].tarColor;
        co2->ObjectsArr[targetSlot].tarScore += 1.0 / co1->numOfObjects;
    }


    // sort co2->ObjectsArr[i2] by score 
    bubbleSort_OD_DetectionItem(co2->ObjectsArr, co2->numOfObjects);


    // trim num objects 
    if(co2->numOfObjects > MAX_TILES_CONSIDER)
        co2->numOfObjects = MAX_TILES_CONSIDER;

    //re-normalize score
    float totalScores = 0;
    for (size_t i2 = 0; i2 < co2->numOfObjects; i2++)
        totalScores = totalScores + co2->ObjectsArr[i2].tarScore;
    for (size_t i2 = 0; i2 < co2->numOfObjects; i2++)
        co2->ObjectsArr[i2].tarScore = co2->ObjectsArr[i2].tarScore/(totalScores+0.00001);


    //TODO: compute scores based on Binomial distribution 
    
}

OD::OD_CycleOutput *GetTargetFromTileImage(const char *imgName, const char *iniFile = NULL)
{
    //TODO: get graph from ini
#ifdef WIN32
    string graph = (char *)"graphs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03_frozen_inference_graph.pb";
#else
    string graph = (char *)"graphs/frozen_inference_graph_humans.pb";
#endif

    uint bigH = 2160;
    uint bigW = 4096;

    //create tiled image
    cv::Mat *bigIm = new cv::Mat(bigH, bigW, CV_8UC3);
    std::list<float *> *tarList = new list<float *>(0);
    CreateTiledImage(imgName, bigW, bigH, bigIm, tarList);

    //run OD on one tiled image
    // Mission
    MB_Mission mission = {
        MB_MissionType::MATMON,                 //mission1.missionType
        e_OD_TargetSubClass::UNKNOWN_SUB_CLASS, //mission1.targetClas
        e_OD_TargetColor::UNKNOWN_COLOR         //mission1.targetColor
    };

    // support data
    OD_SupportData supportData = {
        bigH, bigW, //imageHeight//imageWidth
        //e_OD_ColorImageType::RGB_IMG_PATH,   //colorType;
        e_OD_ColorImageType::RGB, //colorType;
        100,                      //rangeInMeters
        70.0f,                    //fcameraAngle; //BE
        0,                        //TEMP:cameraParams[10];//BE
        0                         //TEMP: float	spare[3];
    };

    OD_InitParams initParams =
        {
            (char *)graph.c_str(),
            350, // max number of items to be returned
            supportData,
            mission};

    // Creation of ATR manager + new mission
    OD::ObjectDetectionManager *atrManager = OD::CreateObjectDetector(&initParams); //first mission
    OD::InitObjectDetection(atrManager, &initParams);
    ((ObjectDetectionManagerHandler *)atrManager)->WaitForThread();

    // Cycles
    OD_CycleInput *ci = new OD_CycleInput();
    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    OD_ErrorCode statusCycle;
    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

    unsigned char *ptrTif = ParseCvMat(*bigIm);
    ci->ptr = ptrTif;

    statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);
    ((ObjectDetectionManagerHandler *)atrManager)->WaitForThread();

    //DEBUG
    atrManager->SaveResultsATRimage(co, "tiles.png", true);

    // analyze results and populate output
    OD_CycleOutput *coOut = new OD_CycleOutput(); // allocate empty cycle output buffer
    AnalyzeTiles(co, tarList, coOut);

    // clean
    delete bigIm;
    std::list<float *>::iterator it;
    for (it = tarList->begin(); it != tarList->end(); ++it)
        delete (*it);
    delete tarList;

    return coOut;
}

int main1()
{
    const char *imname = "media/gzir/gzir001.jpg";
    uint H = 2160;
    uint W = 4096;

    cv::Mat *tiledBeast = new cv::Mat(H, W, CV_8UC3);

    std::list<float *> *tarList = new list<float *>(0);

    CreateTiledImage(imname, W, H, tiledBeast, tarList);

    cv::imwrite("bigTiled.tif", *tiledBeast);
    std::cout << "Num tiles " << tarList->size() << std::endl;

    delete tiledBeast;

    std::list<float *>::iterator it;
    for (it = tarList->begin(); it != tarList->end(); ++it)
        delete (*it);

    tarList->empty();
    delete tarList;

    return 0;
}

int main()
{
    const char *imname = "media/gzir/gzir005.jpg";
    OD::OD_CycleOutput *co_out = GetTargetFromTileImage(imname);

    return 0;
}

int main_future()
{
    const char *imname1 = "media/gzir/gzir001.jpg";
    const char *imname2 = "media/gzir/gzir002.jpg";
    const char *imname3 = "media/gzir/gzir003.jpg";

    #ifdef WIN32
    string graph = (char *)"graphs/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03_frozen_inference_graph.pb";
    #else
    string graph = (char *)"graphs/frozen_inference_graph_humans.pb";
    #endif
    
    // Mission
    MB_Mission mission = {
        MB_MissionType::ANALYZE_SAMPLE,                 //mission1.missionType
        e_OD_TargetSubClass::UNKNOWN_SUB_CLASS, //mission1.targetClas
        e_OD_TargetColor::UNKNOWN_COLOR         //mission1.targetColor
    };
    // support data
    OD_SupportData supportData = {
        0, 0, //imageHeight//imageWidth
        e_OD_ColorImageType::RGB_IMG_PATH,   //colorType;
        //e_OD_ColorImageType::RGB, //colorType;
        100,                      //rangeInMeters
        70.0f,                    //fcameraAngle; //BE
        0,                        //TEMP:cameraParams[10];//BE
        0                         //TEMP: float	spare[3];
    };

    OD_InitParams initParamsSamples =
        {
            (char *)graph.c_str(),
            350, // max number of items to be returned
            supportData,
            mission};
    
    // Creation of ATR manager + new mission
    OD::ObjectDetectionManager *atrManagerSamples = OD::CreateObjectDetector(&initParamsSamples); //first mission
    OD::InitObjectDetection(atrManagerSamples, &initParamsSamples);
    
    // Cycles
    OD_CycleInput *ci = new OD_CycleInput();
    OD_CycleOutput *co = NewOD_CycleOutput(350); // allocate empty cycle output buffer

    OD_ErrorCode statusCycle;

    ci->ptr = (const unsigned char*)imname1;
    statusCycle = OD::OperateObjectDetectionAPI(atrManagerSamples, ci, co);

    ci->ptr = (const unsigned char*)imname2;
    statusCycle = OD::OperateObjectDetectionAPI(atrManagerSamples, ci, co);
    //TODO: print co

    ci->ptr = (const unsigned char*)imname3;
    statusCycle = OD::OperateObjectDetectionAPI(atrManagerSamples, ci, co);
    //TODO: print co


    return 0;
}