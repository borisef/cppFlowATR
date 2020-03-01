//#include "../include/cppflowATR/InterfaceATR.h"
#include "cppflowATR/InterfaceATR.h"
#include <utils/imgUtils.h>
#include <iostream>

mbInterfaceATR::mbInterfaceATR()
{
#ifdef TEST_MODE
    cout << "Construct mbInterfaceATR" << endl;
#endif //TEST_MODE

    m_show = false;
    m_model = nullptr;
    m_outTensorNumDetections = nullptr;
    m_outTensorScores = nullptr;
    m_outTensorBB = nullptr;
    m_outTensorClasses = nullptr;
    m_inpName = nullptr;
}
mbInterfaceATR::~mbInterfaceATR()
{
#ifdef TEST_MODE
    cout << "Destruct mbInterfaceATR" << endl;
#endif //TEST_MODE

    if (m_model != nullptr)
    {
#ifdef TEST_MODE
        cout << "Delete 5 tensors and model" << endl;
#endif //TEST_MODE
        delete m_model;
        delete m_outTensorNumDetections;
        delete m_outTensorScores;
        delete m_outTensorBB;
        delete m_outTensorClasses;
        delete m_inpName;
    }
}

bool mbInterfaceATR::LoadNewModel(const char *modelPath)
{
#ifdef TEST_MODE
    std::cout << " LoadNewModel begin" << std::endl;
#endif //TEST_MODE

    if (m_model != nullptr)
    {
        delete m_model;
        delete m_outTensorNumDetections;
        delete m_outTensorScores;
        delete m_outTensorBB;
        delete m_outTensorClasses;
        delete m_inpName;
    }

    m_model = new Model(modelPath, CreateSessionOptions(0.3));
    m_outTensorNumDetections = new Tensor(*m_model, "num_detections");
    m_outTensorScores = new Tensor(*m_model, "detection_scores");
    m_outTensorBB = new Tensor(*m_model, "detection_boxes");
    m_outTensorClasses = new Tensor(*m_model, "detection_classes");

    m_inpName = new Tensor(*m_model, "image_tensor");

    return true;
}

int mbInterfaceATR::RunRGBimage(cv::Mat inp)
{
    // Put image in Tensor
    std::vector<uint8_t> img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
    m_inpName->set_data(img_data, {1, inp.rows, inp.cols, inp.channels()});

    m_model->run(m_inpName, {m_outTensorNumDetections, m_outTensorScores, m_outTensorBB, m_outTensorClasses});

    inp.copyTo(m_keepImg);
    return 1;
}
int mbInterfaceATR::RunRGBImgPath(const unsigned char *ptr)
{
    cv::Mat inp1 = cv::imread(string((const char *)ptr), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(inp1, inp1, CV_BGR2RGB);

    return RunRGBimage(inp1);
}
int mbInterfaceATR::RunRGBVector(const unsigned char *ptr, int height, int width)
{

#ifdef TEST_MODE
    cout << " RunRGBVector:Internal Run on RGB Vector on ptr*" << endl;
    cout << "RunRGBVector " << height << " " << width << "prt[10]" << ptr[10] << endl;
#endif //TEST_MODE

    std::vector<uint8_t> img_data(height * width * 3);
    unsigned char *buffer = (unsigned char *)ptr;

#ifdef TEST_MODE
    cout << " RunRGBVector:casted buffer to unsigned char* " << endl;
#endif //TEST_MODE

    cv::Mat tempIm(height, width, CV_8UC3);
#ifdef TEST_MODE
    cout << " RunRGBVector:copy buffer to cv::Mat* " << endl;
#endif                    //TEST_MODE
    tempIm.data = buffer; //risky

#ifdef TEST_MODE
    cv::imwrite("tempim.png", tempIm);
#endif //TEST_MODE

    cv::cvtColor(tempIm, tempIm, cv::COLOR_RGB2BGR);
    tempIm.copyTo(m_keepImg);

#ifdef TEST_MODE
    cout << " RunRGBVector:saving cv::Mat* " << endl;
    cv::imwrite("testRGBbuffer.tif", tempIm);
#endif //TEST_MODE

    cv::cvtColor(tempIm, tempIm, cv::COLOR_BGR2RGB); //because we do on original buffer

    for (int i = 0; i < height * width * 3; i++)
        img_data[i] = buffer[i];

    return (RunRGBVector(img_data, height, width));
}
int mbInterfaceATR::RunRGBVector(std::vector<uint8_t> img_data, int height, int width)
{
#ifdef TEST_MODE
    cout << " RunRGBVector:Internal Run on RGB Vector on vector<uint8_t> " << endl;
#endif //TEST_MODE

    // Put image in Tensor
    m_inpName->set_data(img_data, {1, height, width, 3});
    m_model->run(m_inpName, {m_outTensorNumDetections, m_outTensorScores, m_outTensorBB, m_outTensorClasses});
    return 1; //TODO useful return
}
int mbInterfaceATR::RunRawImage(const unsigned char *ptr, int height, int width)
{

    std::vector<uint8_t> img_data(height * width * 2);
    unsigned char *buffer = (unsigned char *)ptr;

    for (int i = 0; i < height * width * 2; i++) //TODO: can we optimize it ?
        img_data[i] = buffer[i];

    //
    cv::Mat *myRGB = new cv::Mat(height, width, CV_8UC3);
    convertYUV420toRGB(img_data, width, height, myRGB);

#ifdef TEST_MODE
    // save JPG for debug
    cv::imwrite("debug_yuv420torgb.tif", *myRGB);
#endif //TEST_MODE

    img_data.assign(myRGB->data, myRGB->data + myRGB->total() * myRGB->channels());
    myRGB->copyTo(m_keepImg);
    delete myRGB; //??? TODO: is it safe?
    int status = RunRGBVector(img_data, height, width);

    return status;
}

int mbInterfaceATR::RunRawImageFast(const unsigned char *ptr, int height, int width, int colorType)
{

    std::vector<uint8_t> img_data(height * width * 2);
    unsigned char *buffer = (unsigned char *)ptr;

    cv::Mat *myRGB = new cv::Mat(height, width, CV_8UC3);

    if (colorType == 7) //NV12
        fastNV12ToRGB((char *)ptr, width, height, myRGB);
    else //YUV422
        fastYUV2RGB((char *)ptr, width, height, myRGB);

#ifdef TEST_MODE
    // save JPG for debug
    cv::imwrite("debug_yuv420torgb.tif", *myRGB);
#endif //TEST_MODE

    img_data.assign(myRGB->data, myRGB->data + myRGB->total() * myRGB->channels());
    myRGB->copyTo(m_keepImg);
    delete myRGB; //??? TODO: is it safe?
    int status = RunRGBVector(img_data, height, width);

    return status;
}

int mbInterfaceATR::GetResultNumDetections()
{
    return (int)m_outTensorNumDetections->get_data<float>()[0];
}

int mbInterfaceATR::GetResultClasses(int i)
{
    return (int)m_outTensorClasses->get_data<float>()[i];
}

float mbInterfaceATR::GetResultScores(int i)
{
    return m_outTensorScores->get_data<float>()[i];
}
std::vector<float> mbInterfaceATR::GetResultBoxes()
{
    return m_outTensorBB->get_data<float>();
}