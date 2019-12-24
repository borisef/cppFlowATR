//#include "../include/cppflowATR/InterfaceATR.h"
#include "cppflowATR/InterfaceATR.h"
#include <utils/imgUtils.h>
#include <iostream>



mbInterfaceATR::mbInterfaceATR()
{
    cout << "Construct mbInterfaceATR" <<endl;
    m_show = false;
    m_model = nullptr;
    m_outTensorNumDetections = nullptr;
    m_outNames2 = nullptr;
    m_outNames3 = nullptr;
    m_outNames4 = nullptr;
    m_inpName  = nullptr;

}
mbInterfaceATR::~mbInterfaceATR()
{
    cout << "Destruct mbInterfaceATR" <<endl;

     if(m_model != nullptr)
    {
        delete m_model;
        delete m_outTensorNumDetections;
        delete m_outNames2;
        delete m_outNames3;
        delete m_outNames4;
        delete m_inpName;
    }
}

bool mbInterfaceATR::LoadNewModel(const char* modelPath)
{
    std::cout<< " LoadNewModel begin" << std::endl;

    if(m_model != nullptr)
    {
        delete m_model;
        delete m_outTensorNumDetections;
        delete m_outNames2;
        delete m_outNames3;
        delete m_outNames4;
        delete m_inpName;
    }

    m_model = new Model(modelPath, CreateSessionOptions( 0.3 ));
    m_outTensorNumDetections = new Tensor(*m_model, "num_detections");
    m_outNames2 = new Tensor(*m_model, "detection_scores");
    m_outNames3 = new Tensor(*m_model, "detection_boxes");
    m_outNames4 = new Tensor(*m_model, "detection_classes");

    m_inpName = new Tensor(*m_model, "image_tensor");



    return true;
}


int mbInterfaceATR::RunRGBimage(cv::Mat inp)
{
    // Put image in Tensor
    std::vector<uint8_t > img_data;
    img_data.assign(inp.data, inp.data + inp.total() * inp.channels());
    m_inpName->set_data(img_data, {1,  inp.rows, inp.cols, inp.channels()});

    m_model->run(m_inpName, {m_outTensorNumDetections, m_outNames2, m_outNames3, m_outNames4});

    return 1;
}

int mbInterfaceATR::RunRGBVector(const unsigned char *ptr, int height, int width)
{

    cout << " RunRGBVector:Internal Run on RGB Vector on ptr*" << endl;


    std::vector<uint8_t > img_data(height*width*3);
    unsigned char* buffer = (unsigned char*)ptr;

    cout << " RunRGBVector:casted buffer to unsigned char* " << endl;

    cv::Mat tempIm(height, width,CV_8UC3);
    cout << " RunRGBVector:copy buffer to cv::Mat* " << endl;
    tempIm.data = buffer; //TODO ???
    cv::cvtColor(tempIm, tempIm, cv::COLOR_RGB2BGR);//TEMP
    cout << " RunRGBVector:saving cv::Mat* " << endl;
    cv::imwrite("testRGBbuffer.tif",tempIm);


    for (int i =0;i<height*width*3;i++)
        img_data[i]=buffer[i];

    return(RunRGBVector(img_data,height,width));
    
}
int mbInterfaceATR::RunRGBVector(std::vector<uint8_t > img_data, int height, int width)
{
    cout << " RunRGBVector:Internal Run on RGB Vector on vector<uint8_t> " << endl;
    // Put image in Tensor
    m_inpName->set_data(img_data, {1,  height, width, 3});
    m_model->run(m_inpName, {m_outTensorNumDetections, m_outNames2, m_outNames3, m_outNames4});
    return 1;

}
int mbInterfaceATR::RunRawImage(const unsigned char *ptr, int height, int width)
{
    
    std::vector<uint8_t > img_data(height*width*3);
    unsigned char* buffer = (unsigned char*)ptr;
   
    for (int i =0;i<height*width*3;i++)//TODO: can we optimize it ? 
        img_data[i]=buffer[i];


     //
    cv::Mat* myRGB = new cv::Mat(height, width,CV_8UC3);
    convertYUV420toRGB(img_data, width, height, myRGB);
  // save JPG for debug
    cv::imwrite("debug_yuv420torgb.tif",*myRGB);
    //std::vector<uint8_t > img_data;
    img_data.assign(myRGB->data, myRGB->data + myRGB->total() * myRGB->channels());
    delete myRGB;//??? TODO: is it safe? 
    int status = RunRGBVector(img_data, height, width);
    
    return status; 

}


int mbInterfaceATR::GetResultNumDetections()
{
    return (int)m_outTensorNumDetections->get_data<float>()[0];
}

int mbInterfaceATR::GetResultClasses(int i)
{
    return (int)m_outNames4->get_data<float>()[i];
}

 float mbInterfaceATR::GetResultScores(int i)
 {
     return m_outNames2->get_data<float>()[i];

 }
std::vector<float>  mbInterfaceATR::GetResultBoxes()
{
    return m_outNames3->get_data<float>();
}