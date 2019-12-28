//by itay

#define VIDEO_PATH "media/00000018.MP4"
#define CONFIG_PATH "config.ini"
#include <opencv2/opencv.hpp>
#include "cppflow/Model.h"
#include "cppflow/Tensor.h"

#include <iostream>
#include <exception>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>

#include "inih/INIReader.h"

void doATRinference(cv::Mat inp1, OD::ObjectDetectionManager *atrManager);
OD_InitParams *loadConfig(string path);
OD::ObjectDetectionManager *createAtrManager();

int main()
{

  cv::Mat frame;
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

    // If the frame is empty, break immediately
    if (frame.empty())
      break;

    //send to tensorflow
    doATRinference(frame, atrManager);

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

OD_InitParams *loadConfig(string path)
{
  INIReader *reader = new INIReader(path);

  if (reader->ParseError() < 0)
  {
    throw "error loading config file \"" + path + "\" is a directory or doesn't exist.";
  }

  std::cout << "Config loaded from 'test.ini': version="
            << reader->GetInteger("protocol", "version", -1)
            << ", name=" << reader->Get("user", "name", "UNKNOWN")
            // << ", email=" << reader->Get("user", "email", "UNKNOWN")
            // << ", pi=" << reader->GetReal("user", "pi", -1)
            // << ", active=" << reader->GetBoolean("user", "active", true)
            << std::endl;

  bool SHOW = false;
  float numIter = 3.0;

  MB_Mission *mission1 = new MB_Mission();
  *mission1 = {
      MB_MissionType::MATMON,       //mission1.missionType
      e_OD_TargetSubClass::PRIVATE, //mission1.targetClas
      e_OD_TargetColor::WHITE       //mission1.targetColor
  };

  // support data
  OD_SupportData *supportData1 = new OD_SupportData();
  *supportData1 = {
      (unsigned int)reader->GetInteger("supportData", "height", 0), //imageHeight
      (unsigned int)reader->GetInteger("supportData", "width", 0),  //imageWidth

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

void doATRinference(cv::Mat inp1, OD::ObjectDetectionManager *atrManager)
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

  // Mission
  OD_InitParams *initParams1 = loadConfig(CONFIG_PATH);

  // Creation of ATR manager + new mission
  OD::ObjectDetectionManager *atrManager = OD::CreateObjectDetector(initParams1); //first mission

  OD::InitObjectDetection(atrManager, initParams1);

  std::cout << " ***  ObjectDetectionManager created  *** " << std::endl;
  return atrManager;
}