//by itay

#define VIDEO_PATH "media/00000018.MP4"

#include <opencv2/opencv.hpp>
#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include <iostream>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>

void doATRinference(cv::Mat inp1, OD::ObjectDetectionManager* atrManager);
OD::ObjectDetectionManager *createAtrManager();

int main()
{
  cv::Mat frame(2160,4096, CV_8UC3);
  cv::VideoCapture cap(VIDEO_PATH);
  OD::ObjectDetectionManager *atrManager = createAtrManager();
  

  // Check if camera opened successfully
  if (!cap.isOpened())
  {
    std::cout << "Error opening video stream or file" << std::endl;
    return -1;
  }

  while (1)
  {
    // Capture frame-by-frame
    frame.release();
    cap >> frame;

    //send to tensorflow
    doATRinference(frame, atrManager);

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    //resize but keep aspect ratio
    cv::resize(frame, frame, cv::Size(frame.size().width / 3, frame.size().height / 3));

    // Display the resulting frame
    //imshow("Frame", frame);

    // Press  ESC on keyboard to exit
    char c = (char)cv::waitKey(25);
    if (c == 27)
      break;
  }

  // When everything done, release the video capture object
  cap.release();
  OD::ObjectDetectionManager *createAtrManager();

  //at the end
  OD::TerminateObjectDetection(atrManager);

  // Closes all the frames
  cv::destroyAllWindows();

  return 0;
}

void doATRinference(cv::Mat inp1, OD::ObjectDetectionManager* atrManager)
{
  


  //emulate buffer from TIF
  cout << " ***  Read tif image to rgb buffer  ***  " << endl;

  cv::cvtColor(inp1, inp1, CV_BGR2RGB);

  //put image in vector
  std::vector<uint8_t> img_data1;
  img_data1.assign(inp1.data, inp1.data + inp1.total() * inp1.channels());

  unsigned char *ptrTif = new unsigned char[img_data1.size()];
  std::copy(begin(img_data1), end(img_data1), ptrTif);

  OD_CycleInput *ci = new OD_CycleInput();
  ci->ImgID_input = 42;
  ci->ptr = ptrTif;

  OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
  OD_ErrorCode statusCycle;

  co->maxNumOfObjects = 300;
  // co->ImgID_output = -1;
  // co->numOfObjects = 300;
  co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];

  cout << " ***  Run inference on RGB image  ***  " << endl;

  statusCycle = OD::OperateObjectDetectionAPI(atrManager, ci, co);

  cout << " ***  .... After inference on RGB image   *** " << endl;

  // //draw
  //cv:Mat* ret = atrManager->ReturnResultsATRimage(ci, co);

  atrManager->SaveResultsATRimage(ci, co, (char *)"out_res2.tif", true);

  //release buffer
  delete ptrTif;

  

  //release OD_CycleInput
  delete ci;

  //release OD_CycleOutput
  delete co->ObjectsArr;
  delete co;

  //return *ret;
}


OD::ObjectDetectionManager *createAtrManager()
{
  bool SHOW = false;
  float numIter = 3.0;

  unsigned int H = 4096;
  unsigned int W = 2160;
  unsigned int frameID = 42;

  // Mission
  MB_Mission *mission1 = new MB_Mission();
  *mission1 = {
      MB_MissionType::MATMON,       //mission1.missionType
      e_OD_TargetSubClass::PRIVATE, //mission1.targetClas
      e_OD_TargetColor::WHITE       //mission1.targetColor
  };

  // support data
  OD_SupportData *supportData1 =  new OD_SupportData();
  *supportData1 = {
      H, W,                     //imageHeight//imageWidth
      e_OD_ColorImageType::RGB, //colorType;
      100,                      //rangeInMeters
      70.0f,                    //fcameraAngle; //BE
      0,                        //TEMP:cameraParams[10];//BE
      0                         //TEMP: float	spare[3];
  };

  OD_InitParams *initParams1 = new OD_InitParams();
  (*initParams1) = {
          //(char*)"/home/magshim/MB2/TrainedModels/faster_MB_140719_persons_sel4/frozen_390k/frozen_inference_graph.pb", //fails
          //(char*)"tryTRT_humans.pb", //sometimes works
          //(char*)"/home/magshim/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb", //works
          //(char*)"tryTRT_humans.pb", //sometimes OK, sometimes crashes the system
          (char *)"graphs/frozen_inference_graph_humans.pb",
          //  (char*)"tryTRT_all.pb", //Nope
          // (char*)"/home/magshim/cppflowATR/frozen_inference_graph_all.pb",
          100, // max number of items to be returned
          *supportData1,
          *mission1};

  // Creation of ATR manager + new mission
  OD::ObjectDetectionManager *atrManager = OD::CreateObjectDetector(initParams1); //first mission

  OD::InitObjectDetection(atrManager, initParams1);

  std::cout << " ***  ObjectDetectionManager created  *** " << std::endl;
  return atrManager;
}