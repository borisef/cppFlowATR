#include "cppflowCM/InterfaceCM.h"
#include <utils/imgUtils.h>
#include <utils/odUtils.h>
#include <iostream>


//constructors/destructors
mbInterfaceCM::mbInterfaceCM()
{
    cout << "Construct mbInterfaceCM" << endl;
    m_model = nullptr;
    m_inTensorPatches = nullptr;
    m_outTensorScores = nullptr;
}
mbInterfaceCM::~mbInterfaceCM()
{
    cout << "Destruct mbInterfaceCM" << endl;

    if (m_model != nullptr)
    {
        delete m_inTensorPatches;
        delete m_outTensorScores;
        delete m_model;
      
    }
}

bool mbInterfaceCM::LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor)
{

    std::cout << " LoadNewModel begin" << std::endl;

    if (m_model != nullptr)
    {
        delete m_model;
        delete m_inTensorPatches;
        delete m_outTensorScores;
    }

    m_model = new Model(modelPath, CreateSessionOptions(0.3));
    if (ckptPath != nullptr)
        m_model->restore(ckptPath);

    m_inTensorPatches = new Tensor(*m_model, intensor);
    m_outTensorScores = new Tensor(*m_model, outtensor);

    return true;
}

std::vector<float> mbInterfaceCM::RunRGBimage(cv::Mat img)
{
    cv::Mat img_resized;
    
    //resize
    cv::resize(img, img_resized, cv::Size(m_patchWidth, m_patchHeight));
    // Put image in vector
    std::vector<float> img_resized_data(m_patchWidth * m_patchHeight * 3);

    for (size_t i = 0; i < img_resized_data.size(); i = i + 1)
    {
        img_resized_data[i] = img_resized.data[i] / 255.0;
    }

    // Put vector in Tensor
    this->m_inTensorPatches->set_data(img_resized_data, {1, img_resized.rows, img_resized.cols, 3});
    //input->set_data(img_resized_data);
    this->m_model->run(m_inTensorPatches, m_outTensorScores);
    return (m_outTensorScores->get_data<float>());
}

std::vector<float> mbInterfaceCM::RunRGBImgPath(const unsigned char *ptr)
{
    cv::Mat img = cv::imread(string((const char*)ptr), CV_LOAD_IMAGE_COLOR);
    return(RunRGBimage(img));
}

void mbInterfaceCM::IdleRun()
{
    if(m_model == nullptr)
        return;
        
    int BS = 1;
    if(m_hardBatchSize > 1)
        BS = m_hardBatchSize;
    std::vector<float> inVec(BS * m_patchHeight * m_patchWidth * 3);
    
    // Put vector in Tensor
    this->m_inTensorPatches->set_data(inVec, {BS, m_patchHeight, m_patchWidth, 3});
    this->m_model->run(m_inTensorPatches, m_outTensorScores);
}

std::vector<float>  mbInterfaceCM::GetResultScores(int i)
{   
    //TODO: check if results do exist 
    
    return (m_outTensorScores->get_data<float>());
}



std::vector<float> mbInterfaceCM::RunImgBB(cv::Mat img, OD::OD_BoundingBox bb)
{
    cv::Mat croppedRef, img_resized;
    cv::Mat debugImg = img.clone();
    //get sub-image

    cv::Rect myROI(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);

    croppedRef = img(myROI);
    cv::rectangle(debugImg, myROI, cv::Scalar(0, 255, 0), 5);
    cv::imwrite("t1.png", img);
    cv::imwrite("t2.png", croppedRef);
    //cv::Mat cropped;
    // Copy the data into new matrix
    //croppedRef.copyTo(cropped);
    //resize
    cv::resize(croppedRef, img_resized, cv::Size(m_patchWidth, m_patchHeight));
    cv::imwrite("t1a.png", debugImg);
    cv::imwrite("t3.png", img_resized);

    //apply
    // Put image in vector
    std::vector<float> img_resized_data(m_patchWidth * m_patchHeight * 3);

    for (size_t i = 0; i < img_resized_data.size(); i = i + 1)
    {
        img_resized_data[i] = img_resized.data[i] / 255.0;
    }

    // Put vector in Tensor
    this->m_inTensorPatches->set_data(img_resized_data, {1, img_resized.rows, img_resized.cols, 3});
    //input->set_data(img_resized_data);
    this->m_model->run(m_inTensorPatches, m_outTensorScores);
    return (m_outTensorScores->get_data<float>());
}

std::vector<float> mbInterfaceCM::RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults)
{
    cv::Mat debugImg = img.clone();

    int N = co->numOfObjects; // N can be smaller or bigger than BS

    int BS = stopInd - startInd + 1;
    if (m_hardBatchSize)
        BS = m_batchSize;
   
    std::vector<float> inVec(BS * m_patchHeight * m_patchWidth * 3);
    int ind = 0;

    for (size_t i = startInd; i <= stopInd; i++)
    {
        if (i >= N)
        {
            //not suppose to happen
            break;
        }
        cv::Mat croppedRef, img_resized;
        //crop
        OD::OD_BoundingBox bb = co->ObjectsArr[i].tarBoundingBox;
        cv::Rect myROI(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);
        //debug
        cv::rectangle(debugImg, myROI, cv::Scalar(0, 255, 0), 5);
        
        croppedRef = img(myROI);
        //resize
        cv::resize(croppedRef, img_resized, cv::Size(m_patchWidth, m_patchHeight));

        // Put image in vector
        for (size_t i1 = 0; i1 < m_patchWidth * m_patchHeight * 3; i1++)
        {
            inVec[ind + i1] = img_resized.data[i1] / 255.0;
        }
        ind += m_patchWidth * m_patchHeight * 3;
    }

    cv::imwrite("color_batch.png", debugImg);

    // Put vector in Tensor
    this->m_inTensorPatches->set_data(inVec, {BS, m_patchHeight, m_patchWidth, 3});

    m_model->run(m_inTensorPatches, m_outTensorScores);
    std::vector<float> allscores = m_outTensorScores->get_data<float>();
    if (copyResults)
    {
        for (size_t si = startInd; si <= stopInd; si++)
        {
            //subvector 
            vector<float>::const_iterator first = allscores.begin() + si * m_numColors;
            vector<float>::const_iterator last = allscores.begin() + (si + 1) * m_numColors;
            vector<float> outRes(first, last);

             //get color 
            //argmax
            uint color_id = std::distance(outRes.begin(), std::max_element(outRes.begin(), outRes.end()));
            
            cout << "color id = " << color_id << endl;
            PrintColor(color_id);
            // score 
            cout << "Net score: " << outRes[color_id] << endl;
            // copy res into co 
            co->ObjectsArr[si].tarColor = TargetColor(color_id);
            co->ObjectsArr[si].tarColorScore = outRes[color_id];
           
        }
    }
    return (allscores);
}

OD::e_OD_TargetColor mbInterfaceCM::TargetColor(uint cid)
{
     switch (cid)
    {
    case 0:
        cout << "Color: white" << endl;
        return OD::e_OD_TargetColor::WHITE;
    case 1:
        cout << "Color: black" << endl;
       return OD::e_OD_TargetColor::BLACK;
    case 2:
        cout << "Color: gray" << endl;
        return OD::e_OD_TargetColor::GRAY;
    case 3:
        cout << "Color: red" << endl;
       return OD::e_OD_TargetColor::RED;
    case 4:
        cout << "Color: green" << endl;
        return OD::e_OD_TargetColor::GREEN;
    case 5:
        cout << "Color: blue" << endl;
        return OD::e_OD_TargetColor::BLUE;
    case 6:
        cout << "Color: yellow" << endl;
        return OD::e_OD_TargetColor::YELLOW;
    default:
        return OD::e_OD_TargetColor::UNKNOWN_COLOR;
    }
    
}
