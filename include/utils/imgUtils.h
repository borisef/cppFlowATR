#include <iostream>
#include <vector>
//#include <e:/Installs/opencv/sources/include/opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

std::vector<unsigned char> readBytesFromFile(const char* filename);
bool convertYUV420toRGB(vector <unsigned char> raw, int width, int height, cv::Mat* outRGB);
bool convertYUV420toVector(vector <unsigned char> raw, int width, int height, std::vector<uint8_t>* outVec );
bool convertCvMatToVector(cv::Mat* inBGR, std::vector<uint8_t>* outVec );

Mat rotate(Mat src, double angle);
cv::Mat CreateTiledGzir(const char* filename, uint W, uint H);


void balance_white(cv::Mat mat);