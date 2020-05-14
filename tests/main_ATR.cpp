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

  
  // MATMON MISSION
  MB_Mission mission1 = {
      MB_MissionType::MATMON,       //mission1.missionType
       e_OD_TargetClass::VEHICLE, //mission1.targetClass
      e_OD_TargetSubClass::PRIVATE, //mission1.targetSubClass
      e_OD_TargetColor::WHITE       //mission1.targetColor
  };

  OD_SupportData supportData1 = {
      H,                       //imageWidth
      W,                       //imageHeight
      e_OD_ColorImageType::COLOR, // colorType;
      100,                        //rangeInMeters
      70.0f,                      //fcameraAngle; //BE
      0,                       //cameraParams[10];//BE
      0                        //float	spare[3];
  };

  OD_InitParams *initParams1 = new OD_InitParams();
  initParams1->iniFilePath = "inifile.txt"; // path to ini file
  initParams1->numOfObjects = 100;          // max number of items to be returned
  initParams1->supportData = supportData1;
  initParams1->mbMission = mission1;

  ObjectDetectionManager *atrManager;
  atrManager = CreateObjectDetector(initParams1); //first mission 

  // new mission
  InitObjectDetection(atrManager, initParams1);


  //emulate buffer from RAW
  std::vector <unsigned char> vv = readBytesFromFile("/home/magshim/cppflowATR/00006160.raw");

  //emulate buffer from tif 
  #ifdef OPENCV_MAJOR_4
  cv::Mat inp = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR
  #else
  cv::Mat inp = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", CV_LOAD_IMAGE_COLOR);
  #endif
  
  cv::cvtColor(inp, inp, CV_BGR2RGB);
  // //put image in vector
  std::vector<uint8_t > img_data;
  img_data.assign(inp.data, inp.data + inp.total() * inp.channels());

  unsigned char *ptr = new unsigned char[vv.size()];
  std::copy(begin(vv), end(vv), ptr);

  OD_CycleInput ci1 = {
    42, //unsigned int ImgID_input;
	  ptr// const unsigned char *ptr; // pointer to picture buffer
  };

  unsigned char *ptr1 = new unsigned char[img_data.size()];
  std::copy(begin(img_data), end(img_data), ptr1);
  OD_CycleInput ci2 = {
    42, //unsigned int ImgID_input;
	  ptr1// const unsigned char *ptr; // pointer to picture buffer
  };


  OD_CycleOutput* co = NULL;

  //TODO: call inference 
  OD_ErrorCode atrStatus = OperateObjectDetectionAPI(atrManager , &ci1 , co );


  //TODO: draw results using  OD_CycleOutput



  //load image 
   // Read image
  //cv::Mat img, inp, imgS;
  //inp = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", CV_LOAD_IMAGE_COLOR);

  // int rows = inp.rows;
  // int cols = inp.cols;
  // cv::cvtColor(inp, inp, CV_BGR2RGB);








  // for loop
  //OperateObjectDetectionAPI()

  bool SHOW = true;
  mbInterfaceATR *mbATR = new mbInterfaceATR();

  mbATR->LoadNewModel("/home/magshim/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb");

  // Read image
  cv:Mat img, imgS;
  #ifdef OPENCV_MAJOR_4
  img = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR
  #else
  img = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", CV_LOAD_IMAGE_COLOR);
  #endif
  int rows = img.rows;
  int cols = img.cols;

  //cv::resize(img, inp, cv::Size(4096, rows));
  cv::cvtColor(img, inp, CV_BGR2RGB);

  mbATR->RunRGBimage(inp);

  float numIter = 5.0;
  auto start = high_resolution_clock::now();
  for (int i = 0; i < numIter; i++)
  {
    std::cout << "Start run *****" << std::endl;
    mbATR->RunRGBimage(inp);
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "*** Duration per detection " << float(duration.count()) / (numIter * 1000000.0f) << " seconds " << endl;

  // Visualize detected bounding boxes.
  int num_detections = mbATR->GetResultNumDetections();
  cout << "***** num_detections " << num_detections << endl;

  for (int i = 0; i < num_detections; i++)
  {
    int classId = mbATR->GetResultClasses(i);
    float score = mbATR->GetResultScores(i);
    auto bbox_data = mbATR->GetResultBoxes();
    std::vector<float> bbox = {bbox_data[i * 4], bbox_data[i * 4 + 1], bbox_data[i * 4 + 2], bbox_data[i * 4 + 3]};
    if (score > 0.1)
    {
      std::cout << "*****" << std::endl;
      float x = bbox[1] * cols;
      float y = bbox[0] * rows;
      float right = bbox[3] * cols;
      float bottom = bbox[2] * rows;
      cv::rectangle(img, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
    }
  }

  if (SHOW)
  {
    cv::resize(img, imgS, cv::Size(1365, 720));
    cv::imshow("Image", imgS);
    cv::waitKey(0);
  }
}
