#include <utils/imgUtils.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace std::chrono;

void stackoverflow_YUV2RGB(void *yuvDataIn, void *rgbDataOut, int w, int h, int outNCh);
void itay_YUV2RGB(char *raw, int width, int height, cv::Mat *outRGB);
void nv12_rgb(char *raw, int height, int width, cv::Mat *outRGB);

char *itay_readBytesFromFile(std::string filepath);
void nv12ToRGBslow(char *raw, int width, int height, cv::Mat *outRGB);

typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template <typename F, typename... Args>
double funcTime(F func, Args &&... args)
{
    TimeVar t1 = timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow() - t1);
}

void NewMain()
{
    {
        int h = 4056;
        int w = 3040;

        char *buf = fastParseRaw("media/00006160.raw");
        cv::Mat rgb(cv::Size(h, w), CV_8UC3);

        cv::Mat *myRGB = new cv::Mat(h, w, CV_8UC3);
        std::vector<uint8_t> img_data(h * w * 2);
        for (int i = 0; i < h * w * 2; i++) //TODO: without for loop
            img_data[i] = buf[i];

        std::cout << "testing boris's convertYUV420toRGB function. " << std::endl;
        std::cout << "time: " << funcTime(convertYUV420toRGB, img_data, h, w, myRGB) << " milliseconds" << std::endl;
        std::cout << "Displaying results... press any key to continue" << std::endl;
        namedWindow("Image RGB", WINDOW_NORMAL);
        imshow("Image RGB", *myRGB);
        waitKey();

        std::cout << "testing itay's YUV2RGB function. " << std::endl;
        std::cout << "time: " << funcTime(fastYUV2RGB, buf, h, w, &rgb) << " milliseconds" << std::endl;
        std::cout << "Displaying results... press any key to continue" << std::endl;

        namedWindow("Image RGB", WINDOW_NORMAL);
        imshow("Image RGB", rgb);
        waitKey();
    }
}

void ItayMain()
{
    int h = 4056;
    int w = 3040;

    char *buf = itay_readBytesFromFile("media/00006160.raw");
    cv::Mat rgb(cv::Size(h, w), CV_8UC3);

    cv::Mat *myRGB = new cv::Mat(h, w, CV_8UC3);
    std::vector<uint8_t> img_data(h * w * 2);
    for (int i = 0; i < h * w * 2; i++) //TODO: without for loop
        img_data[i] = buf[i];

    std::cout << "testing boris's convertYUV420toRGB function. " << std::endl;
    std::cout << "time: " << funcTime(convertYUV420toRGB, img_data, h, w, myRGB) << " milliseconds" << std::endl;
    std::cout << "Displaying results... press any key to continue" << std::endl;
    namedWindow("Image RGB", WINDOW_NORMAL);
    imshow("Image RGB", *myRGB);
    waitKey();

    std::cout << "testing itay's YUV2RGB function. " << std::endl;
    std::cout << "time: " << funcTime(fastYUV2RGB, buf, h, w, &rgb) << " milliseconds" << std::endl;
    std::cout << "Displaying results... press any key to continue" << std::endl;

    namedWindow("Image RGB", WINDOW_NORMAL);
    imshow("Image RGB", rgb);
    waitKey();
}

void nv12main2()
{
    int h = 3040;
    int w = 4056;
    
    char *buf = itay_readBytesFromFile("media/NV12/00000189.raw");
    cv::Mat *rgb = new cv::Mat();
    
    //nv12_rgb(buf, h, w, rgb);
    fastNV12ToRGB(buf,w,h,rgb);;
    

    namedWindow("Image RGB", WINDOW_NORMAL);
    imshow("Image RGB", *rgb);
    waitKey();
}

void NV12Main()
{
    int h = 3040;
    int w = 4056;

    char *buf = itay_readBytesFromFile("media/NV12/00000189.raw");
    cv::Mat rgb(cv::Size(h, w), CV_8UC3);

    cv::Mat *myRGB = new cv::Mat(h, w, CV_8UC3);
    std::vector<uint8_t> img_data(h * w * 2);

    // for (int i = 0; i < h * w * 2; i++) //TODO: without for loop
    //     img_data[i] = buf[i];

    // std::cout << "testing boris's convertYUV420toRGB function. " << std::endl;
    // std::cout << "time: " << funcTime(convertYUV420toRGB, img_data, h, w, myRGB) << " milliseconds" << std::endl;
    // std::cout << "Displaying results... press any key to continue" << std::endl;
    // namedWindow("Image RGB", WINDOW_NORMAL);
    // imshow("Image RGB", *myRGB);
    // waitKey();
    
   
    nv12ToRGBslow(buf, w,h, myRGB);
    namedWindow("Image RGB", WINDOW_NORMAL);
    imshow("Image RGB", *myRGB);
    waitKey();
}


char *itay_readBytesFromFile(std::string filepath)
{
    std::ifstream file(filepath, ios::binary | ios::ate);
    ifstream::pos_type pos = file.tellg();

    int length = pos;

    if (length == 0)
    {
        cout << "length = 0. error reading file" << endl;
        return nullptr;
    }

    char *buffer = new char[length];
    file.seekg(0, std::ios::beg);
    file.read(buffer, length);

    file.close();

    return buffer;
}

void itay_YUV2RGB(char *raw, int height, int width, cv::Mat *outRGB)
{
    cv::Mat yuyv442(cv::Size(height, width), CV_8UC2, raw);
    cvtColor(yuyv442, *outRGB, COLOR_YUV2RGB_YUYV);
}

void nv12ToRGBslow(char *raw, int width, int height, cv::Mat *outRGB)
{
    cv::Mat Y(cv::Size( width,height), CV_8UC1, raw);
    cv::Mat U(cv::Size( width/2,height/2), CV_8UC1);
    cv::Mat V(cv::Size( width/2,height/2), CV_8UC1);

    int t = width*height;
    for (int i = 0; i < height/2; i++)
        for (int j = 0; j < width/2; j++)
    {

      U.at<uint8_t>(i, j) = raw[t ];
      V.at<uint8_t>(i, j) = raw[t + 1];
      t = t + 2;
    }
 

  cv::resize(U,U,cv::Size(width,height));
  cv::resize(V,V,cv::Size(width,height));
  
  cv::imwrite("tryY_NV12.png", Y);
  cv::imwrite("tryU_NV12.png", U);
  cv::imwrite("tryV_NV12.png", V);

  cv::Mat yuv;

  std::vector<cv::Mat> yuv_channels = {Y, U, V};
  cv::merge(yuv_channels, yuv);

  // cv::Mat rgb(height, width,CV_8UC3);
  cv::cvtColor(yuv, *outRGB, cv::COLOR_YUV2BGR);
  cv::imwrite("RGB_NV12.tif", *outRGB);



   
}

void nv12_rgb(char *raw, int height, int width, cv::Mat *outRGB)
{
    cv::Mat nv12(height * 3/2, width, CV_8UC1, raw);
    cvtColor(nv12, *outRGB, CV_YUV2RGB_NV12);
}

int main(int argc, char const *argv[])
{
    nv12main2();

    return 0;
}
