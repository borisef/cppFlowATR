#include <iostream>
#include <vector>
//#include <e:/Installs/opencv/sources/include/opencv2/opencv.hpp>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

std::vector<unsigned char> readBytesFromFile(const char* filename);
bool convertYUV420toRGB(vector <unsigned char> raw, int width, int height, cv::Mat* outRGB);
inline void fastYUV2RGB(char *raw, int height, int width, cv::Mat *outRGB)
{
    cv::Mat yuyv442(cv::Size(height, width), CV_8UC2, raw);
    cvtColor(yuyv442, *outRGB, COLOR_YUV2RGB_YUYV);
}
bool nv12ToRGB(char *raw, int width, int height, cv::Mat *outRGB);
inline void fastNV12ToRGB(char *raw,  int width, int height, cv::Mat *outRGB)
{
    cv::Mat nv12(height * 3/2, width, CV_8UC1, raw);
    cvtColor(nv12, *outRGB, COLOR_YUV2RGB_NV12);//CV_YUV2RGB_NV12=90
}


bool convertYUV420toVector(vector <unsigned char> raw, int width, int height, std::vector<uint8_t>* outVec );
bool convertCvMatToVector(cv::Mat* inBGR, std::vector<uint8_t>* outVec );


Mat rotate(Mat src, double angle);
bool CreateTiledImage(const char* filename, uint W, uint H, cv::Mat* bigImg, list <float*> *trueTargets);


void balance_white(cv::Mat mat);

unsigned char *ParseImage(String path);
unsigned char *ParseRaw(String path);
unsigned char *ParseCvMat(cv::Mat inp1);
float IoU(float* , float*);

uint argmax_vector(std::vector<float> prob_vec);

inline bool file_exists_test (const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

char *fastParseRaw(std::string filepath);

