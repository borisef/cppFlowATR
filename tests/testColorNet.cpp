#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>

#define PATCH_WIDTH 128
#define PATCH_HEIGHT 128

using namespace std;
using namespace std::chrono;

void PrintColor(int color_id)
{
    switch (color_id)
    {
    case 0:
        cout << "Color: white" << endl;
        break;
    case 1:
        cout << "Color: black" << endl;
        break;
    case 2:
        cout << "Color: gray" << endl;
        break;
    case 3:
        cout << "Color: red" << endl;
        break;
    case 4:
        cout << "Color: green" << endl;
        break;
    case 5:
        cout << "Color: blue" << endl;
        break;
    case 6:
        cout << "Color: yellow" << endl;
        break;
    }
}
uint argmax_color(std::vector<float> prob_vec)
{
    float max_val = 0;
    uint index = 0;
    uint argmax = 0;
    for (std::vector<float>::iterator it = prob_vec.begin(); it != prob_vec.end(); ++it)
    {
        if (max_val < *it)
        {
            max_val = *it;
            argmax = index;
        }
        index++;
    }

    return argmax;
}

int main()
{
    const char *imges[9] = {"media/color/color001.png",
                            "media/color/color002.png",
                            "media/color/color003.png",
                            "media/color/color004.png",
                            "media/color/color005.png",
                            "media/color/color006.png",
                            "media/color/color007.png",
                            "media/color/color008.png",
                            "media/color/color009.png"};

    // Display mode
    bool flag_display = false;

    // Model
    //Model model("e:/projects/MB/ColorNitzan/TFexample/outColorNetOutputs_30_01_20/frozen/frozen_cmodel.pb");
    //Model model("e:/projects/MB/ColorNitzan/TFexample/model2/tf_model2.pb");
    Model model("e:/projects/MB/ColorNitzan/TFexample/output_graph.pb");

    // Model model("e:/projects/MB/ColorNitzan/TFexample/outColorNetOutputs_30_01_20/model/model.pb");
    // model.restore("e:/projects/MB/ColorNitzan/TFexample/outColorNetOutputs_30_01_20/model/checkpoint/train.ckpt");

    // Input tensor
    auto input = new Tensor(model, "conv2d_input");
    // Output tensor
    auto output = new Tensor(model, "dense_1/Softmax");

    // Read imagep
    cv::Mat img, img_resized;
    //img = cv::Mat::zeros(PATCH_WIDTH, PATCH_HEIGHT, CV_8UC3); // Choose any size you want
    //img.setTo(cv::Scalar(0, 200.0, 0.0));
    // Do it several times
    for (size_t sample = 0; sample < 9; sample++)
    {
        auto start = high_resolution_clock::now();
        //img.setTo(cv::Scalar(0.0, 255.0, 0.0));
        img = cv::imread(imges[sample], CV_LOAD_IMAGE_COLOR);

        // Manipulate image
        //cv::cvtColor(img, img, CV_BGR2RGB);
        cv::resize(img, img_resized, cv::Size(PATCH_WIDTH, PATCH_HEIGHT));

        //img_resized.convertTo(img_resized, CV_32F);

        // Put image in vector
        std::vector<float> img_resized_data(PATCH_WIDTH * PATCH_HEIGHT * 3);

        //img_resized_data.assign(img_resized.data, img_resized.data + img_resized.total() * img_resized.channels());

        for (size_t i = 0; i < img_resized_data.size(); i = i + 1)
        {
            img_resized_data[i] = img_resized.data[i] / 255.0;
        }

        // Put vector in Tensor
        input->set_data(img_resized_data, {1, img_resized.rows, img_resized.cols, 3});
        //input->set_data(img_resized_data);
        model.run(input, output);
        uint color_id = argmax_color(output->get_data<float>());
        cout << "color id = " << color_id << endl;

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        cout << "*** Duration per detection " << float(duration.count()) / (1 * 1000000.0f) << " seconds " << endl;
        // {0: 'white', 1: 'black', 2: 'gray', 3: 'red', 4: 'green', 5: 'blue', 6: 'yellow'}
        PrintColor(color_id);
        cout << "Net score: " << output->get_data<float>()[color_id] << endl;

        if (flag_display)
        {
            cv::imshow("Image", img_resized);
            cv::waitKey(0);
        }
    }

    int BS = 9;
    // Put image in vector
    std::vector<float> batch_img_resized_data(BS * PATCH_WIDTH * PATCH_HEIGHT * 3);
    int ind = 0;
    //try batch
    for (size_t sample = 0; sample < BS; sample++)
    {
        img = cv::imread(imges[sample], CV_LOAD_IMAGE_COLOR);
        cv::resize(img, img_resized, cv::Size(PATCH_WIDTH, PATCH_HEIGHT));

        //img_resized_data.assign(img_resized.data, img_resized.data + img_resized.total() * img_resized.channels());

        for (size_t i = 0; i < PATCH_WIDTH * PATCH_HEIGHT * 3; i = i + 1)
        {
            batch_img_resized_data[ind] = img_resized.data[i] / 255.0;
            ind++;
        }
    }
    // Put vector in Tensor
    input->set_data(batch_img_resized_data, {BS, PATCH_HEIGHT, PATCH_WIDTH, 3});
    //input->set_data(img_resized_data);
    model.run(input, output);
    std::vector<float> res = output->get_data<float>();
    for (size_t si = 0; si < BS; si++)
    {
        vector<float>::const_iterator first = res.begin() + si * 7;
        vector<float>::const_iterator last = res.begin() + (si + 1) * 7 ;
        vector<float> outRes(first, last);
        uint color_id = argmax_color(outRes);
        cout << "color id = " << color_id << endl;
        PrintColor(color_id);
        cout << "Net score: " << outRes[color_id] << endl;
    }

    return 0;
}
