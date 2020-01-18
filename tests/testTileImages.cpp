#include <utils/imgUtils.h>

#include <opencv2/opencv.hpp>
//#include <e:/Installs/opencv/sources/include/opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace std::chrono;
using namespace cv;
//using namespace OD;
//Mat rotate(Mat src, double angle);
RNG rng(12345);

int main()
{
    const char* imname = "media/gzir/gzir001.jpg";
    uint H = 2000;
    uint W = 4000;
   
    cv::Mat tiledBeast = CreateTiledGzir(imname, W, H);

    return 0; 

}



