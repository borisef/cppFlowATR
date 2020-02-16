#include <utils/imgUtils.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace std::chrono;

void stackoverflow_YUV2RGB(void *yuvDataIn, void *rgbDataOut, int w, int h, int outNCh);
void itay_YUV2RGB(char *raw, int width, int height, cv::Mat *outRGB);
void nv12_rgb(char *raw, int height, int width, cv::Mat *outRGB);

char *itay_readBytesFromFile(std::string filepath);

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

void nv12main()
{
    int w = 4056;
    int h = 3040;

    char *buf = itay_readBytesFromFile("media/nv12/00000920.raw");
    cv::Mat *rgb = new cv::Mat();
    
    nv12_rgb(buf, h, w, rgb);

    namedWindow("Image RGB", WINDOW_NORMAL);
    imshow("Image RGB", *rgb);
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

void nv12_rgb(char *raw, int height, int width, cv::Mat *outRGB)
{
    cv::Mat nv12(height * 3/2, width, CV_8UC1, raw);
    cvtColor(nv12, *outRGB, CV_YUV2RGB_YV12);
}

int main(int argc, char const *argv[])
{
    nv12main();

    return 0;
}
