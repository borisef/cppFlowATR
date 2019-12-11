//
// Created by sergio on 16/05/19.
// Changed be
//

#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include "cppflowATR/InterfaceATR.h"

#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono> 


using namespace std; 
using namespace std::chrono;

int main() {


//ObjectDetectionManager* nm = CreateObjectDetector(); // before everything 


///
//InitObjectDetection() // new mission 


// for loop 
//OperateObjectDetectionAPI()





















    bool SHOW = true;
    mbInterfaceATR* mbATR = new mbInterfaceATR();


    mbATR->LoadNewModel("/home/borisef/projects/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb");

    // Read image
    cv::Mat img, inp, imgS;
    img = cv::imread("/home/borisef/projects/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", CV_LOAD_IMAGE_COLOR);

    int rows = img.rows;
    int cols = img.cols;

	
    cv::resize(img, inp, cv::Size(4096,2160));
    cv::cvtColor(inp, inp, CV_BGR2RGB);

    mbATR->RunRGBimage(inp);    

    float numIter = 5.0;
    auto start = high_resolution_clock::now(); 
    for (int i=0;i<numIter;i++){
      std::cout << "Start run *****" << std::endl;  
      mbATR->RunRGBimage(inp);  
    }
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << "*** Duration per detection " << float(duration.count())/(numIter*1000000.0f) << " seconds "<<  endl;

    // Visualize detected bounding boxes.
    int num_detections = mbATR->GetResultNumDetections();
    cout << "***** num_detections " << num_detections << endl;
   

    for (int i=0; i<num_detections; i++) {
           int classId = mbATR->GetResultClasses(i);
           float score = mbATR->GetResultScores(i);
           auto bbox_data = mbATR->GetResultBoxes();
           std::vector<float> bbox = {bbox_data[i*4], bbox_data[i*4+1], bbox_data[i*4+2], bbox_data[i*4+3]};
         if (score > 0.1) {
	         std::cout << "*****" << std::endl;
             float x = bbox[1] * cols;
             float y = bbox[0] * rows;
             float right = bbox[3] * cols;
             float bottom = bbox[2] * rows;
             cv::rectangle(img, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
         }
    }
	
    if(SHOW){
    cv::resize(img, imgS, cv::Size(1365, 720)) ;
    cv::imshow("Image", imgS);
    cv::waitKey(0);}
}
