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
Mat rotate(Mat src, double angle);
RNG rng(12345);

int main()
{
    const char* imname = "media/gzir/gzir001.jpg";
    uint H = 2000;
    uint W = 4000;
    uint gapX=100, gapY = 100;
    bool flag = true;
    uint maxX = 0;
    uint maxY = 0;
    uint topX = 10;
    uint topY = 10;

    cv::Mat bigImg(H, W, CV_8UC3);
    cv::Mat im = cv::imread(imname, CV_LOAD_IMAGE_COLOR);
    //Scalar v( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    Scalar v( 100, 100, 100 );

    int top = max(im.cols - im.rows, 0)/2 + 10;
    int lef = max(im.rows - im.cols, 0)/2 + 10;


    copyMakeBorder( im, im, top, top,  lef, lef, BORDER_CONSTANT, v );



    uint nextYline = 0;
    double angle = 0;

    while (flag)
    {
        if(topX+im.cols>W - gapX)
            {
                topX = 10;
                topY = nextYline + gapY;

            }
        
        if(topY+im.cols>H-gapY)
            break;
    
        Mat im1 = rotate(im,angle);
        if(angle<90)
         balance_white(im1);

        im1.copyTo(bigImg(cv::Rect(topX+rand()%gapX,topY + rand()%gapY,im.cols, im.rows)));
        nextYline = max(nextYline, topY + im.rows);
        topX = topX + im.cols + gapX;
        angle++;
    }
    cv::imwrite("try.tif", bigImg);



}



Mat rotate(Mat src, double angle)
{
    Mat dst;
    Point2f pt(src.cols/2., src.rows/2.);    
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}