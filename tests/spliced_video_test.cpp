#define CONFIG_PATH "config.ini"

#include <iostream>
#include <exception>

#include <opencv2/opencv.hpp>
#include "cppflow/Model.h"
#include "cppflow/Tensor.h"

#include <cppflowATRInterface/Object_Detection_API.h>
#include <cppflowATRInterface/Object_Detection_Types.h>
#include <utils/imgUtils.h>

#include "inih/INIReader.h"

int main()
{
    INIReader *reader = new INIReader(CONFIG_PATH);
    string path = reader->Get("spliced_video", "path", "") + "/*";

    cv::vector<String> fn;
    cv::glob(path, fn, true);
    for (size_t i = 0; i < fn.size(); i++)
    {
        cv::Mat frame = imread(fn[i]);
        cv::resize(frame, frame, cv::Size(frame.size().width / 3, frame.size().height / 3));
        cv::imshow("Frame", frame);

        char c = (char)cv::waitKey(25);
        if (c == 27)
            break;
    }
}
