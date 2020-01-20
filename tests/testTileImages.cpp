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

void AnalyzeTiles(OD_CycleOutput *co1, std::list<float*> *tarList, OD_CycleOutput * co2)
{
    co2->maxNumOfObjects = 100;
    co2->numOfObjects = 0;
    co2->ObjectsArr = new OD_DetectionItem[co2->maxNumOfObjects];


    //temp 
    
    for (size_t i = 0; i < co1->numOfObjects; i++)
    {
        cout<<co1->ObjectsArr[i].tarClass<<endl;
        cout<<co1->ObjectsArr[i].tarSubClass<<endl;
        cout<<co1->ObjectsArr[i].tarColor<<endl;

        // if already exists increment score

        // if not add element co2->numOfObjects, co2->numOfObjects++
    }
    




}

OD::OD_CycleOutput* GetTargetFromGzir(const char* imgName, const char* iniFile = NULL)
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
    cv::Mat* bigIm= new cv::Mat(bigH, bigW, CV_8UC3);
    std::list<float*> *tarList = new list<float*>(0);
    CreateTiledGzir(imgName, bigW, bigH, bigIm, tarList);
    
    //run OD on one tiled image 
     // Mission
    MB_Mission mission = {
        MB_MissionType::ANALYZE_SAMPLE,       //mission1.missionType
        e_OD_TargetSubClass::UNKNOWN_SUB_CLASS, //mission1.targetClas
        e_OD_TargetColor::UNKNOWN_COLOR       //mission1.targetColor
    };

    // support data
    OD_SupportData supportData = {
        bigH, bigW, //imageHeight//imageWidth
        //e_OD_ColorImageType::RGB_IMG_PATH,   //colorType;
        e_OD_ColorImageType::RGB,   //colorType;
        100,          //rangeInMeters
        70.0f,        //fcameraAngle; //BE
        0,            //TEMP:cameraParams[10];//BE
        0             //TEMP: float	spare[3];
    };

    OD_InitParams initParams =
        {
            (char *)graph.c_str(),
            350, // max number of items to be returned
            supportData,
            mission};

    // Creation of ATR manager + new mission
    OD::ObjectDetectionManager *  atrManager = OD::CreateObjectDetector(&initParams); //first mission
    OD::InitObjectDetection(atrManager, &initParams);
    ((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();
        
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
    ((ObjectDetectionManagerHandler*)atrManager)->WaitForThread();

    //DEBUG
    atrManager->SaveResultsATRimage(co, "tiles.png", true);
    
    //TODO: analyze results and populate output
    OD_CycleOutput *coOut = new OD_CycleOutput(); // allocate empty cycle output buffer
    AnalyzeTiles(co, tarList, coOut);
     

    // clean 
    delete bigIm; 
    std::list<float*>::iterator it;
    for (it=tarList->begin(); it!=tarList->end(); ++it)
        delete (*it);
    delete tarList;

    return co;

}




int main1()
{
    const char* imname = "media/gzir/gzir001.jpg";
    uint H = 2160;
    uint W = 4096;
   
    cv::Mat* tiledBeast= new cv::Mat(H, W, CV_8UC3);
    
    std::list<float*> *tarList = new list<float*>(0);
   

    CreateTiledGzir(imname, W, H, tiledBeast, tarList);

    cv::imwrite("bigTiled.tif", *tiledBeast);
    std::cout<<"Num tiles "<<tarList->size()<<std::endl;

    delete tiledBeast;
    
    std::list<float*>::iterator it;
    for (it=tarList->begin(); it!=tarList->end(); ++it)
        delete (*it);

    tarList->empty();
    delete tarList;
    




    return 0; 

}



int main()
{
    const char* imname = "media/gzir/gzir005.jpg";
    OD::OD_CycleOutput* co_out = GetTargetFromGzir(imname);


    return 0;
}



