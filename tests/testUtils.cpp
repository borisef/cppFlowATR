#include <utils/imgUtils.h>

#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;
using namespace cv;
//using namespace OD;



int main()
{
  int H = 4056;
  int W = 3040;

  cout << "Test utils" << endl;
  //emulate buffer
  cout << "readBytesFromFile raw" << endl;
  std::vector <unsigned char> rawVec = readBytesFromFile("00006160.raw");
  
  //
  cv::Mat* myRGB = new Mat(H, W,CV_8UC1);
  convertYUV420toRGB(rawVec, H, W, myRGB);
  // save JPG for debug

  //DEBUG
  cv::Mat bgr(H, W,CV_8UC1);
  cv::cvtColor(*myRGB, bgr, cv::COLOR_RGB2BGR);
  cv::imwrite("raw2im.tif", bgr);
  
  // RAW -> RGB vector to be used in inference
  std::vector <unsigned char>* myVector;
  convertYUV420toVector(rawVec, H, W, myVector);

  //Read TIF -> mat -> RGB vector to be used in  inference
  cv::Mat im  = cv::imread("orig.tif", CV_LOAD_IMAGE_COLOR);
  std::vector <unsigned char>* myVector1;
  convertCvMatToVector(&im, myVector1); 

  return 0; 
}