#include <cppflowATRInterface/Object_Detection_Handler.h>
#include <utils/imgUtils.h>

#include <iomanip>
#include <future>

using namespace OD;
using namespace std;

OD_CycleInput *NewCopyCycleInput(OD_CycleInput *tocopy, uint bufferSize)
{
    OD_CycleInput *newCopy = new OD_CycleInput();
    newCopy->ImgID_input = tocopy->ImgID_input;
    //copy buffer
    unsigned char *buffer = new unsigned char[bufferSize];
    memcpy(buffer, tocopy->ptr, bufferSize);
    newCopy->ptr = buffer;
    return newCopy;
}

OD_CycleInput *SafeNewCopyCycleInput(OD_CycleInput *tocopy, uint bufferSize)
{
    if (!tocopy)
        return nullptr;
    OD_CycleInput *newCopy = new OD_CycleInput();
    newCopy->ImgID_input = tocopy->ImgID_input;
    //copy buffer
    if (tocopy->ptr)
    {
        unsigned char *buffer = new unsigned char[bufferSize];
        memcpy(buffer, tocopy->ptr, bufferSize);
        newCopy->ptr = buffer;
    }
    else
    {
        newCopy->ptr = nullptr;
    }

    return newCopy;
}

void DeleteCycleInput(OD_CycleInput *todel)
{
    if (todel)
    {
        if (todel->ptr)
            delete todel->ptr;
        delete todel;
    }
}

OD_InitParams *ObjectDetectionManagerHandler::getParams() { return m_initParams; }

void ObjectDetectionManagerHandler::setParams(OD_InitParams *ip)
{
    m_initParams = ip;
    m_numPtrPixels = ip->supportData.imageHeight * ip->supportData.imageWidth;
    m_numImgPixels = ip->supportData.imageHeight * ip->supportData.imageWidth * 3;
    if (ip->supportData.colorType == e_OD_ColorImageType::YUV422)
        m_numPtrPixels = m_numPtrPixels * 2;
    else
        m_numPtrPixels = m_numPtrPixels * 3;
}

bool ObjectDetectionManagerHandler::IsBusy()
{
    if (m_result.valid())
    {   
        bool temp = !(m_result.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
        if(temp)
            cout<<" IsBusy? Yes"<<endl;
        else
        {
            cout<<" IsBusy? No"<<endl;
        }
        
        return !(m_result.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    }
    else
    {
        cout<<" IsBusy? Not valid"<<endl;
        return false;
    }
}

ObjectDetectionManagerHandler::ObjectDetectionManagerHandler()
{
    m_mbATR = nullptr;
    m_prevCycleInput = nullptr;
    m_nextCycleInput = nullptr;
    m_curCycleInput = nullptr;
}
ObjectDetectionManagerHandler::ObjectDetectionManagerHandler(OD_InitParams *ip) : ObjectDetectionManagerHandler()
{
    setParams(ip);
}

ObjectDetectionManagerHandler::~ObjectDetectionManagerHandler()
{
    WaitForThread();
    DeleteAllInnerCycleInputs();

    if (m_mbATR != nullptr)
        delete m_mbATR;
}

bool ObjectDetectionManagerHandler::WaitForThread()
{
    if (m_result.valid())
    {
        std::cout << "waiting...\n";
        m_result.wait();
        std::cout << "Done!\n";
    }
    else
    {
        cout << "Thread is still not valid" << endl;
    }

    return true;
}

bool ObjectDetectionManagerHandler::WaitUntilForThread(int sec)
{
    if (m_result.valid())
    {
        std::chrono::system_clock::time_point few_seconds_passed = std::chrono::system_clock::now() + std::chrono::seconds(sec);

        if (std::future_status::ready == m_result.wait_until(few_seconds_passed))
        {
            std::cout << "times_out "
                      << "\n";
        }
        else
        {
            std::cout << "did not complete!\n";
        }

        std::cout << "Done!\n";
    }
    else
    {
        cout << "Thread is still not valid" << endl;
    }

    return true;
}
void ObjectDetectionManagerHandler::DeleteAllInnerCycleInputs()
{
    if (m_nextCycleInput)
    {
        DeleteCycleInput(m_nextCycleInput);
        m_nextCycleInput = nullptr;
    }
    if (m_curCycleInput)
    {
        DeleteCycleInput(m_curCycleInput);
        m_curCycleInput = nullptr;
    }
    if (m_prevCycleInput)
    {
        DeleteCycleInput(m_prevCycleInput);
        m_prevCycleInput = nullptr;
    }
}

OD_ErrorCode ObjectDetectionManagerHandler::InitObjectDetection(OD_InitParams *odInitParams)
{
    //check if busy, if yes wait till the end and assign nullptr to next, current and prev
    WaitForThread();
    DeleteAllInnerCycleInputs();

    mbInterfaceATR *mbATR = nullptr;
    //initialization
    if (m_mbATR == nullptr)
    {
        mbATR = new mbInterfaceATR();
        cout << "Create new mbInterfaceATR in ObjectDetectionManagerHandler::InitObjectDetection" << endl;
        //TODO: decide which model to take (if to take or stay with old )
        mbATR->LoadNewModel(odInitParams->iniFilePath);
        m_mbATR = mbATR;
        cout << "Executed LoadNewModel in  InitObjectDetection" << endl;
    }

    setParams(odInitParams);

    return OD_ErrorCode::OD_OK;
}
OD_ErrorCode ObjectDetectionManagerHandler::PrepareOperateObjectDetection(OD_CycleInput *cycleInput)
{
    //keep OD_CycleInput copy
    cout << " PrepareOperateObjectDetection: Prepare to run on frame " << cycleInput->ImgID_input << endl;
    if (cycleInput && cycleInput->ptr) // input is valid
    {

        OD_CycleInput *tempCycleInput = nullptr;
        cout << "Replace old next cycle input" << endl;
       m_mutexOnNext.lock();
        if (m_nextCycleInput) //my next not null
        {
            if (cycleInput->ImgID_input != m_nextCycleInput->ImgID_input)
            { // not same frame#
                //replace next
                tempCycleInput = m_nextCycleInput;

                m_nextCycleInput = NewCopyCycleInput(cycleInput, m_numPtrPixels);

                DeleteCycleInput(tempCycleInput); //delete old "next"
            }
            else // same next frame
            {
                cout << "PrepareOperateObjectDetection: attempt to call with same frame twice, skipping" << endl;
            }
        }
        else
        { // m_nextCycleInput is null, safe to create it
            m_nextCycleInput = NewCopyCycleInput(cycleInput, m_numPtrPixels);
        }
        m_mutexOnNext.unlock();
    }
    else
    {
        cout << "PrepareOperateObjectDetection:Input cycle is null, skip " << endl;
    }
    if (m_nextCycleInput && m_nextCycleInput->ptr)
        cout << "PrepareOperateObjectDetection:Next cycle is valid" << endl;
    else
        cout << "PrepareOperateObjectDetection:Next cycle is empty" << endl;

    return OD_ErrorCode::OD_OK;
}

OD_ErrorCode ObjectDetectionManagerHandler::OperateObjectDetection(OD_CycleOutput *odOut)
{

    cout << "^^^Locked" << endl;

    if (m_nextCycleInput && m_nextCycleInput->ptr)
        cout << "OperateObjectDetection:Next cycle is valid" << endl;
    else
        cout << "OperateObjectDetection:Next cycle is empty" << endl;

    if (m_curCycleInput && m_curCycleInput->ptr)
        cout << "OperateObjectDetection:Current cycle is valid" << endl;
    else
        cout << "OperateObjectDetection:Current cycle is empty" << endl;

    if (m_prevCycleInput && m_prevCycleInput->ptr)
        cout << "OperateObjectDetection:Previous cycle is valid" << endl;
    else
        cout << "OperateObjectDetection:Previous cycle is empty" << endl;

    if (m_curCycleInput) // never suppose to happen, jic
    {
        cout << "This was never  supposed to happen... m_curCycleInput is not empty... How? " << endl;
        DeleteCycleInput(m_curCycleInput);
        m_curCycleInput = nullptr;
        return OD_ErrorCode::OD_FAILURE;
    }
    // deep copy next into current, next is not null here
   m_mutexOnNext.lock();
    m_curCycleInput = SafeNewCopyCycleInput(m_nextCycleInput, m_numPtrPixels); // safe take care of NULL
    DeleteCycleInput(m_nextCycleInput); //just to allow recursive call of OperateObjectDetection
    m_nextCycleInput = nullptr; 
    m_mutexOnNext.unlock();

    if (!(m_curCycleInput && m_curCycleInput->ptr)) // former next (current ) cycle or buffer is null
    {
        cout << "ObjectDetectionManagerHandler::OperateObjectDetection nothing todo" << endl;
        cout << "###UnLocked" << endl;
        DeleteCycleInput(m_curCycleInput);
        m_curCycleInput = nullptr;

        return OD_ErrorCode::OD_OK;
    }

    cout << " OperateObjectDetection: Run on frame " << m_curCycleInput->ImgID_input << endl;

    unsigned int fi = m_curCycleInput->ImgID_input;
    int h = m_initParams->supportData.imageHeight;
    int w = m_initParams->supportData.imageWidth;

    e_OD_ColorImageType colortype = m_initParams->supportData.colorType;

    if (colortype == e_OD_ColorImageType::YUV422) // if raw
        this->m_mbATR->RunRawImage(m_curCycleInput->ptr, h, w);
    else if (colortype == e_OD_ColorImageType::RGB) // if rgb
    {
        cout << " Internal Run on RGB buffer " << endl;
        this->m_mbATR->RunRGBVector(m_curCycleInput->ptr, h, w);
    }
    else
    {
        return OD_ErrorCode::OD_ILEGAL_INPUT;
    }
    cout << " ObjectDetectionManagerHandler::OperateObjectDetection starts PopulateCycleOutput  " << endl;
    // save results
    this->PopulateCycleOutput(odOut);
    odOut->ImgID_output = fi;

    // copy current into prev
    cout << "ObjectDetectionManagerHandler::OperateObjectDetection m_prevCycleInput<=m_curCycleInput<=nullptr" << endl;
    OD_CycleInput *tempCI = m_prevCycleInput;
    m_prevCycleInput = m_curCycleInput; // transfere

   m_mutexOnPrev.lock();
    DeleteCycleInput(tempCI);
    m_mutexOnPrev.unlock();

    m_curCycleInput = nullptr;

    // string outName = "outRes/out_res3_" + std::to_string(odOut->ImgID_output) + "_in.png";
    // this->SaveResultsATRimage(nullptr,odOut, (char*)outName.c_str(),false);

    cout << "###UnLocked" << endl;
    
    //Recoursive call 
    if(m_nextCycleInput) 
    {
        cout<<"+++++++++++++++++++++++++++++++++Call OperateObjectDetection again automatically"<<endl;
        OperateObjectDetection(odOut);
    }
    return OD_ErrorCode::OD_OK;
}

bool ObjectDetectionManagerHandler::SaveResultsATRimage(OD_CycleOutput *co, char *imgNam, bool show)
{
    OD_CycleInput *tempci = nullptr;
    m_mutexOnPrev.lock();
    if (!(m_prevCycleInput && m_prevCycleInput->ptr))
    {
        cout << "No m_prevCycleInput data, skipping" << endl;
        return false;
    }
    else
    {
        // deep copy m_prevCycleInput 
        tempci = NewCopyCycleInput(m_prevCycleInput, this->m_numPtrPixels);
    }
    m_mutexOnPrev.unlock();

    float drawThresh = 0; //if 0 draw all
    //TODO: make sure m_prevCycleInput->ImgID_input is like co->ImgID
    unsigned int fi = tempci->ImgID_input;
    unsigned int h = m_initParams->supportData.imageHeight;
    unsigned int w = m_initParams->supportData.imageWidth;

    e_OD_ColorImageType colortype = m_initParams->supportData.colorType;
    cv::Mat *myRGB = nullptr;
    unsigned char *buffer = (unsigned char *)(tempci->ptr);

    if (colortype == e_OD_ColorImageType::YUV422) // if raw
    {
        std::vector<uint8_t> img_data(h * w * 2);
        for (int i = 0; i < h * w * 2; i++) //TODO: without for loop
            img_data[i] = buffer[i];

        myRGB = new cv::Mat(h, w, CV_8UC3);
        convertYUV420toRGB(img_data, w, h, myRGB);
    }
    else if (colortype == e_OD_ColorImageType::RGB) // if rgb
    {
        myRGB = new cv::Mat(h, w, CV_8UC3);
        //myRGB->data = buffer;// NOT safe if we going to use buffer later
        std::copy(buffer, buffer + w * h * 3, myRGB->data);
        //std::copy(arrayOne, arrayOne+10, arrayTwo);
        cv::imwrite("debug_newImg1_bgr.tif", *myRGB);
    }
    else
    {
        DeleteCycleInput(tempci);
        return false;
    }

    std::cout << "***** num_detections " << co->numOfObjects << std::endl;
    for (int i = 0; i < co->numOfObjects; i++)
    {
        int classId = co->ObjectsArr[i].tarClass;
        float score = co->ObjectsArr[i].tarScore;
        OD_BoundingBox bbox_data = co->ObjectsArr[i].tarBoundingBox;

        std::vector<float> bbox = {bbox_data.x1, bbox_data.x2, bbox_data.y1, bbox_data.y2};

        if (score >= drawThresh)
        {
            cout << "add rectangle to drawing" << endl;
            float x = bbox_data.x1;
            float y = bbox_data.y1;
            float right = bbox_data.x2;
            float bottom = bbox_data.y2;

            cv::rectangle(*myRGB, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
            cv::putText(*myRGB, string("Label:") + std::to_string(classId) + ";" + std::to_string(int(score * 100)) + "%", cv::Point(x, y - 10), 1, 2, Scalar(124, 200, 10), 3);
        }
    }
    cout << " Done reading targets" << endl;
    if (show)
    {
        cv::Mat imgS;
        cv::resize(*myRGB, imgS, cv::Size(1365, 720));
        cv::imshow("Image", imgS);

        char c = (char)cv::waitKey(25);
       
        // cv::waitKey(0);
    }
    cv::Mat bgr(h, w, CV_8UC3);
    cv::cvtColor(*myRGB, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(imgNam, bgr);
    cout << " Done saving image" << endl;
    if (myRGB != nullptr)
    {
        myRGB->release();
        delete myRGB; 
    }
    cout << " Done cleaning image" << endl;
    DeleteCycleInput(tempci);
    return true;
}

int ObjectDetectionManagerHandler::PopulateCycleOutput(OD_CycleOutput *cycleOutput)
{
    float LOWER_SCORE_THRESHOLD = 0.1;
    cout << "ObjectDetectionManagerHandler::PopulateCycleOutput" << endl;

    OD_DetectionItem *odi = cycleOutput->ObjectsArr;

    cycleOutput->numOfObjects = m_mbATR->GetResultNumDetections();

    auto bbox_data = m_mbATR->GetResultBoxes();
    unsigned int w = this->m_initParams->supportData.imageWidth;
    unsigned int h = this->m_initParams->supportData.imageHeight;
    cycleOutput->numOfObjects = cycleOutput->maxNumOfObjects;
    for (int i = 0; i < cycleOutput->maxNumOfObjects; i++)
    {
        e_OD_TargetClass aa = e_OD_TargetClass(1);
        odi[i].tarClass = e_OD_TargetClass(m_mbATR->GetResultClasses(i));
        odi[i].tarScore = m_mbATR->GetResultScores(i);
        if (odi[i].tarScore < LOWER_SCORE_THRESHOLD)
        {
            cycleOutput->numOfObjects = i;
            break;
        }

        odi[i].tarBoundingBox = {bbox_data[i * 4 + 1] * w, bbox_data[i * 4 + 3] * w, bbox_data[i * 4] * h, bbox_data[i * 4 + 2] * h};
    }

    return cycleOutput->numOfObjects;
}