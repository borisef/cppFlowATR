//#include "../include/cppflowATR/InterfaceATR.h"
#include "cppflowATR/InterfaceATR.h"



mbInterfaceATR::mbInterfaceATR()
{
    m_show = false;

}

bool mbInterfaceATR::LoadNewModel(const char* modelPath)
{
    m_model = new Model(modelPath);



    m_outNames1 = new Tensor(*m_model, "num_detections");
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
    m_inpName->set_data(img_data, {1,  2160, 4096, 3});

    m_model->run(m_inpName, {m_outNames1, m_outNames2, m_outNames3, m_outNames4});

    return 1;
}

int mbInterfaceATR::GetResultNumDetections()
{
    return (int)m_outNames1->get_data<float>()[0];
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