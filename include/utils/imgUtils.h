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

// The MIT License (MIT)
// Copyright (c) 2015 tomykaira
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// reference http://www.ipol.im/pub/art/2011/llmps-scb/
void balance_white(cv::Mat mat);