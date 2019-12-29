#include <utils/imgUtils.h>

using namespace cv;
using namespace std;

std::vector<unsigned char> readBytesFromFile(const char* filename)
{
    std::vector<unsigned char> result;

    FILE* f = fopen(filename, "rb");

    fseek(f, 0, SEEK_END);  // Jump to the end of the file
    long length = ftell(f); // Get the current byte offset in the file
    rewind(f);              // Jump back to the beginning of the file

    result.resize(length);

    char* ptr = reinterpret_cast<char*>(&(result[0]));
    fread(ptr, length, 1, f); // Read in the entire file
    fclose(f); // Close the file

    return result;
}

bool convertYUV420toRGB(vector <unsigned char> raw, int width, int height, cv::Mat* outRGB)//,
                        // vector <unsigned char>* R = NULL, 
                        // vector <unsigned char>* G = NULL,
                        // vector <unsigned char>* B = NULL)
    {
    // vector <unsigned char> Y ; Y.reserve(width*height);
    // vector <unsigned char> U ; U.reserve(width*height);
    // vector <unsigned char> V ; V.reserve(width*height);
    //TODO: this is very slow
    cv:Mat ymat(height, width,CV_8UC1), umat(height, width,CV_8UC1), vmat(height, width,CV_8UC1);
   
    int t = 0;
    int skip = 1;
    for (int i=0;i<height;i++)
        for(int j = 0;j<width;j++)
        {
        // Y[t]=raw[t*2];
       
      
        ymat.at<uint8_t>(i,j)=raw[t*2];
        if(skip >0)
        {
        // U[2*t]=raw[t*2+1];
        // U[2*t+1]=U[2*t];
        // V[2*t]=raw[t*2+3];
        // V[2*t+1]=V[2*t];
        
        umat.at<uint8_t>(i,j)=raw[t*2 + 1];
        umat.at<uint8_t>(i,j+1)=raw[t*2 + 1];
        
        vmat.at<uint8_t>(i,j)=raw[t*2 + 3];
        vmat.at<uint8_t>(i,j+1)=raw[t*2 + 3];
        }
        skip = -skip;
        t = t + 1;

    }
    cv::imwrite("tryY.png",ymat);
    cv::imwrite("tryU.png",umat);
    cv::imwrite("tryV.png",vmat);
    
    cv::Mat yuv;

    std::vector<cv::Mat> yuv_channels = { ymat, umat, vmat };
    cv::merge(yuv_channels, yuv);

    // cv::Mat rgb(height, width,CV_8UC3);
    cv::cvtColor(yuv, *outRGB, cv::COLOR_YUV2RGB);
    cv::imwrite("RGB.tif", *outRGB);

return true;
}



bool convertYUV420toVector(vector <unsigned char> raw, int width, int height, vector <uint8_t>* outVector)
    {

    cv:Mat ymat(height, width,CV_8UC1), umat(height, width,CV_8UC1), vmat(height, width,CV_8UC1);
   
    int t = 0;
    int skip = 1;
    for (int i=0;i<height;i++)
        for(int j = 0;j<width;j++)
        {
        // Y[t]=raw[t*2];
       
      
        ymat.at<uint8_t>(i,j)=raw[t*2];
        if(skip >0)
        {
        umat.at<uint8_t>(i,j)=raw[t*2 + 1];
        umat.at<uint8_t>(i,j+1)=raw[t*2 + 1];
        
        vmat.at<uint8_t>(i,j)=raw[t*2 + 3];
        vmat.at<uint8_t>(i,j+1)=raw[t*2 + 3];
        }
        skip = -skip;
        t = t + 1;

    }
    cv::imwrite("tryY.png",ymat);
    cv::imwrite("tryU.png",umat);
    cv::imwrite("tryV.png",vmat);
    
    cv::Mat yuv;

    std::vector<cv::Mat> yuv_channels = { ymat, umat, vmat };
    cv::merge(yuv_channels, yuv);

    cv::Mat rgb(height, width,CV_8UC3);
    cv::cvtColor(yuv, rgb, cv::COLOR_YUV2RGB);
   
    outVector = new std::vector<uint8_t >();
    outVector->reserve(width*height*3);
    int a = outVector->size();
    outVector->assign(rgb.data, rgb.data + rgb.total() * rgb.channels());

return true;
}

bool convertCvMatToVector(cv::Mat* inBGR, std::vector<uint8_t>* outVec )
{
    int h = inBGR->rows;
    int w = inBGR->cols;
    int c = inBGR->channels();

    cv::Mat rgb(h, w, CV_8UC3);
    cv::cvtColor(*inBGR, rgb, cv::COLOR_BGR2RGB);

    outVec = new std::vector<uint8_t >();
    outVec->reserve(w*h*c);
    outVec->assign(rgb.data, rgb.data + rgb.total() * rgb.channels());

    return true;
}