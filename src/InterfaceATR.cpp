//#include "../include/cppflowATR/InterfaceATR.h"
#include "cppflowATR/InterfaceATR.h"
#include <utils/imgUtils.h>



mbInterfaceATR::mbInterfaceATR()
{
    m_show = false;

}

bool mbInterfaceATR::LoadNewModel(const char* modelPath)
{
    m_model = new Model(modelPath);



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
    std::vector<uint8_t > img_data(height*width*3);
    unsigned char* buffer = (unsigned char*)ptr;

    for (int i =0;i<height*width*3;i++)
        img_data[i]=buffer[i];

    return(RunRGBVector(img_data,height,width));
    
}
int mbInterfaceATR::RunRGBVector(std::vector<uint8_t > img_data, int height, int width)
{
    // Put image in Tensor
    m_inpName->set_data(img_data, {1,  height, width, 3});
    m_model->run(m_inpName, {m_outTensorNumDetections, m_outNames2, m_outNames3, m_outNames4});
    return 1;

}
int mbInterfaceATR::RunRawImage(const unsigned char *ptr, int height, int width)
{
    
    std::vector<uint8_t > img_data(height*width*3);
    unsigned char* buffer = (unsigned char*)ptr;
   
    for (int i =0;i<height*width*3;i++)
        img_data[i]=buffer[i];


     //
    cv::Mat* myRGB = new cv::Mat(height, width,CV_8UC1);
    convertYUV420toRGB(img_data, height, width, myRGB);
  // save JPG for debug
    //std::vector<uint8_t > img_data;
    img_data.assign(myRGB->data, myRGB->data + myRGB->total() * myRGB->channels());
    
    int status = RunRGBVector(img_data, height, width);
    //std::vector<uint8_t >::size_type size = strlen((const char*)buffer);
    //std::vector<uint8_t > vec(buffer, size + buffer);

    //m_inpName->set_data(img_data, {1,  height, width, 3});
    //m_model->run(m_inpName, {m_outTensorNumDetections, m_outNames2, m_outNames3, m_outNames4});
    return 0;

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