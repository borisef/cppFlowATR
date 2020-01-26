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
#include <algorithm>
#include <iterator>

using namespace std;
using namespace std::chrono;

int main()
{
  // Create model
  Model m("temp/model.pb");
  m.restore("temp/checkpoint/train.ckpt");

  // Create Tensors
  auto input = new Tensor(m, "input");
  auto prediction = new Tensor(m, "prediction");

  // Read image
  for (int i = 0; i < 10; i++)
  {
    cv::Mat img;
    cv::Mat img1;
    cv::Mat scaled;

    // Read image
    img = cv::imread("e:/projects/MB2/cppFlowATR/temp/images/" + std::to_string(i) + ".png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    // Scale image to range 0-1
    img.convertTo(scaled, CV_64F, 1.f / 255);
    
    // Put image in vector
    std::vector<double> img_data;
      //img_data.assign(scaled.begin<double>(), scaled.end<double>());
    //img_data.assign(img.data, img.data + img.total()*img.channels() );


    if (img.isContinuous())
    {
      img_data.assign((double *)scaled.data, (double *)scaled.data + scaled.total());
    }
    else
    {
      for (int i = 0; i < img.rows; ++i)
      {
        img_data.insert(img_data.end(), scaled.ptr<float>(i), scaled.ptr<float>(i) + scaled.cols);
      }
    }

    // Feed data to input tensor
    input->set_data(img_data);

    // Run and show predictions
    m.run(input, prediction);

    // Get tensor with predictions
    auto result = prediction->get_data<double>();

    // Maximum prob
    auto max_result = std::max_element(result.begin(), result.end());

    // Print result
    std::cout << "Real label: " << i << ", predicted: " << std::distance(result.begin(), max_result)
              << ", Probability: " << (*max_result) << std::endl;
  }

  return 0;
}