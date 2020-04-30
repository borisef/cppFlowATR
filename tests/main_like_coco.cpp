//
// Created by sergio on 16/05/19.
// Changed be
//

#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono> 


using namespace std; 
using namespace std::chrono;

int main() {
    //Model model("../ssd_inception/frozen_inference_graph.pb");
    //Model model("/home/magshim/MB2/TrainedModels/MB3_persons_likeBest1_default/frozen_378K/frozen_inference_graph.pb");
    Model model("frozen_inference_graph.pb");
    auto outNames1 = new Tensor(model, "num_detections");
    auto outNames2 = new Tensor(model, "detection_scores");
    auto outNames3 = new Tensor(model, "detection_boxes");
    auto outNames4 = new Tensor(model, "detection_classes");

    auto inpName = new Tensor(model, "image_tensor");

    bool SHOW = false;
    // Read image
    cv::Mat img, inp, imgS;
    //img = cv::imread("/home/magshim/MB2/test_videos/magic_box-test_060519/11.8-sortie_1-clip_16_frames/00000018.tif", CV_LOAD_IMAGE_COLOR);
    img = cv::imread("00000018.tif", IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR

    int rows = img.rows;
    int cols = img.cols;
	
    //cv::resize(img, inp, cv::Size(4096,2160));
    cv::cvtColor(img, inp, CV_BGR2RGB);

    //put image in vector
    std::vector<uint8_t > img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
    
    // Put VECTOR in Tensor
    inpName->set_data(img_data, {1,  rows, cols, 3});
    model.run(inpName, {outNames1, outNames2, outNames3, outNames4});

    float numIter = 3.0;
    auto start = high_resolution_clock::now(); 
    for (int i=0;i<numIter;i++){
    	std::cout << "Start run *****" << std::endl;  
	inpName->set_data(img_data, {1,  2160, 4096, 3});
    	model.run(inpName, {outNames1, outNames2, outNames3, outNames4});
    }
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    cout << "*** Duration per detection " << float(duration.count())/(numIter*1000000.0f) << " seconds "<<  endl;

    // Visualize detected bounding boxes.
    int num_detections = (int)outNames1->get_data<float>()[0];
    std::cout << "***** num_detections " << num_detections << std::endl;    
    for (int i=0; i<num_detections; i++) {
        int classId = (int)outNames4->get_data<float>()[i];
        float score = outNames2->get_data<float>()[i];
        auto bbox_data = outNames3->get_data<float>();
        std::vector<float> bbox = {bbox_data[i*4], bbox_data[i*4+1], bbox_data[i*4+2], bbox_data[i*4+3]};
        // std::cout << "*****" << std::endl;
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

return 0;

}
