//
// Created by sergio on 16/05/19.
// Changed be
//

#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include "cppflowATR/InterfaceATR.h"
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
      H,W,                       //imageHeight//imageWidth
      e_OD_ColorImageType::YUV422, // colorType;
      100,                        //rangeInMeters 
      70.0f,                      //fcameraAngle; //BE
      NULL,                       //cameraParams[10];//BE
      NULL                        //float	spare[3];
  };

  // ATR params
  OD_InitParams *initParams1 = new OD_InitParams();
  initParams1->iniFilePath =  "inifile.txt"; // path to ini file
  initParams1->numOfObjects = 100;          // max number of items to be returned
  initParams1->supportData = supportData1;
  initParams1->mbMission = mission1;

  // Creation of ATR manager + new mission
  ObjectDetectionManager *atrManager;
  atrManager = CreateObjectDetector(initParams1); //first mission 

  // new mission
  InitObjectDetection(atrManager, initParams1);


  //emulate buffer from RAW
  std::vector <unsigned char> vecFromRaw = readBytesFromFile("/home/borisef/projects/cppflowATR/00006160.raw");

  unsigned char *ptr = new unsigned char[vecFromRaw.size()];
  std::copy(begin(vecFromRaw), end(vecFromRaw), ptr);

  OD_CycleInput * ci1 = new OD_CycleInput(frameId);
  c1->ptr = ptr;

  OD_CycleOutput* co = new OD_CycleOutput(initParams1->numOfObjects); // allocate empty cycle output buffer 
  OD_ErrorCode statusCycle;
  statusCycle = OperateObjectDetectionAPI(atrManager , ci1 , co);

  //TODO: draw 
  //atrManager->SaveResultsATRimage(ci1,co,"res1.tif");


  //TODO: run on image from file
  OD_SupportData supportData2 = {
      2160, 4096,                       //imageHeight//imageWidth
      e_OD_ColorImageType::RGB, // colorType;
      100,                        //rangeInMeters 
      70.0f,                      //fcameraAngle; //BE
      NULL,                       //cameraParams[10];//BE
      NULL                        //float	spare[3];
  };
 initParams1->supportData = supportData2;
 // new mission
  InitObjectDetection(atrManager, initParams1);

  //TODO: draw
  //atrManager->SaveResultsATRimage(ci1,co,"res2.tif");
  
  
  

}
