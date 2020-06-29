#include "cppflow/Model.h"
#include "cppflow/Tensor.h"
#include "cppflowCM/InterfaceCM.h"
#include "utils/imgUtils.h"
#include "utils/odUtils.h"
#include <opencv2/opencv.hpp>
#include <numeric>
#include <iomanip>
#include <chrono>

#define PATCH_WIDTH 128
#define PATCH_HEIGHT 128

using namespace std;
using namespace std::chrono;

OD::OD_CycleOutput* CreateSynthCO(cv::Mat img, uint numBB);


int mainWithInterface()
{
    const char *modelPath = "graphs/cm/output_graph_08_03_20.pb";
    const char *ckpt = nullptr;
    const char *inname = "conv2d_input_3";
    const char *outname = "dense_1_1/Softmax";

    const char *inimage = "media/00000018.tif";
    const char *smallim= "media/color/color001.png";

    OD::OD_BoundingBox sampleBB = OD::OD_BoundingBox({100, 200, 150, 220});

    mbInterfaceCM *myCM = new mbInterfaceCM();
    if (!myCM->LoadNewModel(modelPath, ckpt, inname, outname))
    {
        std::cout << "ooops" << std::endl;
        return -1;
    }

    //load BIG image
    #ifdef OPENCV_MAJOR_4
    cv::Mat bigImg = cv::imread(inimage,  IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR
    #else
    cv::Mat bigImg = cv::imread(inimage,  CV_LOAD_IMAGE_COLOR);//CV_LOAD_IMAGE_COLOR
    #endif
    //create synthetic cycleoutput
    OD::OD_CycleOutput* co = CreateSynthCO(bigImg, 10);
    //load small image
     #ifdef OPENCV_MAJOR_4
     cv::Mat smallImg = cv::imread(smallim,IMREAD_COLOR); // CV_LOAD_IMAGE_COLOR
     #else
    cv::Mat smallImg = cv::imread(smallim,CV_LOAD_IMAGE_COLOR); // 
    #endif



    //test on patch
    std::vector<float> vecScores = myCM->RunImgBB(bigImg, sampleBB);
    uint color_id = argmax_vector(vecScores);
    cout << "color id = " << color_id << endl;
    PrintColor(color_id);
    cout << "Net score: " << vecScores[color_id] << endl;

    //test on batch
    bool flag = myCM->RunImgWithCycleOutput(bigImg, co, 0, (co->numOfObjects -1), true);
    
    //test on img name 
    vecScores = myCM->RunRGBImgPath((const uchar*)inimage);
    color_id = argmax_vector(vecScores);
    cout << "color id = " << color_id << endl;
    PrintColor(color_id);
    cout << "Net score: " << vecScores[color_id] << endl;

    //test on mat
    vecScores = myCM->RunRGBimage(smallImg);
    color_id = argmax_vector(vecScores);
    cout << "color id = " << color_id << endl;
    PrintColor(color_id);
    cout << "Net score: " << vecScores[color_id] << endl;
    return 0;
}

int main()
{
    mainWithInterface();

    return 0;
}

int testMain()
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
        #ifdef OPENCV_MAJOR_4
        img = cv::imread(imges[sample], IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR
        #else
        img = cv::imread(imges[sample], CV_LOAD_IMAGE_COLOR);//
        #endif
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
        uint color_id = argmax_vector(output->get_data<float>());
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
        #ifdef OPENCV_MAJOR_4
        img = cv::imread(imges[sample], IMREAD_COLOR);//CV_LOAD_IMAGE_COLOR=1
        #else
        img = cv::imread(imges[sample], CV_LOAD_IMAGE_COLOR);//=1
        #endif
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
        vector<float>::const_iterator last = res.begin() + (si + 1) * 7;
        vector<float> outRes(first, last);
        uint color_id = argmax_vector(outRes);
        cout << "color id = " << color_id << endl;
        PrintColor(color_id);
        cout << "Net score: " << outRes[color_id] << endl;
    }

    return 0;
}




OD::OD_CycleOutput* CreateSynthCO(cv::Mat img, uint numBB)
{

    OD::OD_CycleOutput* co = new OD::OD_CycleOutput();
    co->maxNumOfObjects = 350;
    co->ImgID_output = 0;
    co->numOfObjects = numBB;
    co->ObjectsArr = new OD::OD_DetectionItem[co->maxNumOfObjects];

    int x = 10;
    int y = 10;
    int w = 100;
    int h = 70;
    int step = int((float)img.rows/((float)numBB+1));

    for (size_t i = 0; i < numBB; i++)
    {
        co->ObjectsArr[i].tarBoundingBox = OD::OD_BoundingBox({(float)x, (float)x+w, (float)y, (float)y+h});

        x = x + step;
        y = y + step;
    }

    return co;
}