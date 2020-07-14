//
// 1) Run on thousands of images with different delays
// 2) Create and destroy managers
//

#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <thread>

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>
#include <cppflowATRInterface/Object_Detection_Handler.h>

#include <bits/stdc++.h> 
#include <iostream> 
#include <sys/stat.h> 
#include <sys/types.h> 





using namespace std;
using namespace std::chrono;
using namespace OD;

using std::string;

string getFileName(const string& s) {

   char sep = '/';

#ifdef _WIN32
   sep = '\\';
#endif

   size_t i = s.rfind(sep, s.length());
   if (i != string::npos) {
      return(s.substr(i+1, s.length() - i));
   }

   return("");
}

struct OneRunStruct
{
    int H;
    int W;
    string splicePath;
    string outputPath;
    
    e_OD_ColorImageType imType = e_OD_ColorImageType::NV12;
};


vector<String> GetFileNames(const char *pa)
{

    vector<String> fn;
    cv::glob(pa, fn, true);

    return fn;
}


int OneRunConvertAndSave( OneRunStruct ors)
{
    vector<String> ff = GetFileNames((char *)ors.splicePath.c_str());
    int N = ff.size();
    unsigned char * ptrTif;


    // Creating a directory 
    if (mkdir(ors.outputPath.c_str(), 0777) == -1) 
        cerr << "Error :  " << strerror(errno) << endl; 
    else
        cout << "Directory created/"; 


    for (size_t i = 10; i < N-10; i = i + 3)
    {
        try
        {
         ptrTif = (unsigned char*)fastParseRaw(ff[i]);
         std::vector<uint8_t> img_data(int(ors.H ) * int(ors.W)  * 2 ); 
         cv::Mat *myRGB = new cv::Mat(ors.H, ors.W, CV_8UC3);

         fastNV12ToRGB((char *)ptrTif, ors.W, ors.H, myRGB);
         cv::cvtColor(*myRGB, *myRGB, cv::COLOR_RGB2BGR);
         string fn = getFileName(ff[i]);
         string outfn = std::string(ors.outputPath);//.resize().append(".tif");
         outfn.append(fn);
         outfn.resize(outfn.size() - 4);
         outfn = outfn.append(".tif");
         cv::imwrite(outfn , *myRGB);
         cout<< fn << "  --> " << outfn <<endl;
         delete myRGB;
        }
        catch (int e)
        {
            cout << "An exception occurred. Exception Nr. " << e << '\n';
        }
    }
    
  return N; 
}

int main()
{
    OneRunStruct ors2;
   
    ors2.W = 4056;
    ors2.H = 3040;

    
    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_092033_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_092033_XX_tif/";
    // OneRunConvertAndSave(ors2);

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_115024_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_115024_XX_tif/";
    // OneRunConvertAndSave(ors2);

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_125002_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_125002_XX_tif/";
    // OneRunConvertAndSave(ors2);

   

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_150015_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_150015_XX_tif/";
    // OneRunConvertAndSave(ors2);

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_160029_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_160029_XX_tif/";
    // OneRunConvertAndSave(ors2);

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_170917_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_170917_XX_tif/";
    // OneRunConvertAndSave(ors2);

    // ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_085834_XX/";
    // ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_085834_XX_tif/";
    // OneRunConvertAndSave(ors2);

    ors2.splicePath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_133943_XX/";
    ors2.outputPath = "/mnt/d1e28558-1377-4bbb-9e48-c8900feaf59d/lachish_day1/20200709_133943_XX_tif/";
    OneRunConvertAndSave(ors2);

    cout << "Ended OneRunConvertAndSave Normally" << endl;
    return 0;
}
