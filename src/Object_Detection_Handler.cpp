#include <cppflowATRInterface/Object_Detection_Handler.h>
#include <utils/imgUtils.h>
#include <utils/odUtils.h>

#include <iomanip>
#include <future>

using namespace OD;
using namespace std;

mbInterfaceCM *ObjectDetectionManagerHandler::m_mbCM = nullptr;
InitParams *ObjectDetectionManagerHandler::m_configParams = nullptr;

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
    else if (ip->supportData.colorType == e_OD_ColorImageType::NV12)
        m_numPtrPixels = m_numPtrPixels * 1.5;
    else
        m_numPtrPixels = m_numPtrPixels * 3;
}

bool ObjectDetectionManagerHandler::IsBusy()
{
    if (m_result.valid())
    {
        bool temp = !(m_result.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
#ifdef TEST_MODE
        if (temp)
            cout << " IsBusy? Yes" << endl;
        else
        {
            cout << " IsBusy? No" << endl;
        }
#endif //#ifdef TEST_MODE
        return !(m_result.wait_for(std::chrono::seconds(0)) == std::future_status::ready);
    }
    else
    {
#ifdef TEST_MODE
        cout << " IsBusy? Not valid" << endl;
#endif //#ifdef TEST_MODE
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
#ifdef TEST_MODE
        std::cout << "waiting...\n";
#endif //TEST_MODE

        m_result.wait();
#ifdef TEST_MODE
        std::cout << "Done!\n";
#endif //TEST_MODE
    }
    else
    {
#ifdef TEST_MODE
        cout << "Thread is still not valid" << endl;
#endif //TEST_MODE
    }

    return true;
}

// bool ObjectDetectionManagerHandler::WaitUntilForThread(int sec)
// {
//     if (m_result.valid())
//     {
//         std::chrono::system_clock::time_point few_seconds_passed = std::chrono::system_clock::now() + std::chrono::seconds(sec);

//         if (std::future_status::ready == m_result.wait_until(few_seconds_passed))
//         {
//             std::cout << "times_out "<< std::endl;
//         }
//         else
//         {
//             std::cout << "did not complete!\n";
//         }

//         std::cout << "Done!\n";
//     }
//     else
//     {
//         cout << "Thread is still not valid" << endl;
//     }

//     return true;
// }

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

std::string ObjectDetectionManagerHandler::DefinePathForATRModel()
{

    std::string prepath = m_configParams->run_params["prePath"];
    std::string mo = m_configParams->models[0]["load_path"];

    // use m_configParams and m_initParams to get model path
    for (size_t i = 0; i < m_configParams->models.size(); i++)
    {
#ifdef TEST_MODE
        std::cout << m_configParams->models[i]["nickname"] << std::endl;
        std::cout << m_configParams->models[i]["load_path"] << std::endl;
#endif //TEST_MODE

        if (m_configParams->models[i]["nickname"].compare("default_ATR") == 0)
        {
            mo = (m_configParams->models[i]["load_path"]);
            break;
        }
    }
    prepath.append(mo).append("\0");
    return prepath;
}

OD_ErrorCode ObjectDetectionManagerHandler::InitObjectDetection(OD_InitParams *odInitParams)
{
    //initialization
    //take care of InitParams
    if (m_configParams != nullptr)
        if (m_configParams->GetFilePath().compare(odInitParams->iniFilePath) != 0) //replace
        {
            delete m_configParams;
            m_configParams = nullptr;
        }
    if (m_configParams == nullptr)
        m_configParams = new InitParams(odInitParams->iniFilePath);

    //TODO: if something wrong with init params return error

    // take care of log file
    if (!logInitialized)
    {
        InitializeLogger();
        logInitialized = true;
    }
    //From now on can use logger !!!

    LOG_F(INFO, "Manager initialized from %s", odInitParams->iniFilePath);

    //check if busy, if yes wait till the end and assign nullptr to next, current and prev
    WaitForThread();
    DeleteAllInnerCycleInputs();

    mbInterfaceATR *mbATR = nullptr;

    // define path for ATR model
    std::string pathATR = DefinePathForATRModel();
    LOG_F(INFO, "Defined path for ATR model %s", pathATR.c_str());

    //ATR model initialization
    if (m_mbATR != nullptr)
        if (m_lastPathATR.compare(pathATR) != 0)
        {
            delete m_mbATR;
            m_mbATR = nullptr;
        }
    if (m_mbATR == nullptr)
    {
        mbATR = new mbInterfaceATR();
#ifdef TEST_MODE
        cout << "Create new mbInterfaceATR in ObjectDetectionManagerHandler::InitObjectDetection" << endl;
#endif //TEST_MODE
        mbATR->LoadNewModel(pathATR.c_str());
        m_mbATR = mbATR;
        LOG_F(INFO, "Executed LoadNewModel in  InitObjectDetection");
    }

    //remember lastPathATR
    m_lastPathATR = std::string(pathATR);

    if (odInitParams->supportData.colorType == e_OD_ColorImageType::RGB_IMG_PATH)
    {
        odInitParams->supportData.imageHeight = 2048;
        odInitParams->supportData.imageWidth = 4096;
    }
    //Color Model initialization
    bool initCMsuccess = true;

    if (m_mbCM == nullptr && m_withActiveCM)
    {
        initCMsuccess = InitCM();
        m_withActiveCM = initCMsuccess;
    }

    setParams(odInitParams);

#ifdef TEST_MODE
    LOG_F(INFO, "Do IdleRun() for ATR");
#endif //#ifdef TEST_MODE

    IdleRun();
    if (m_mbCM != nullptr && m_withActiveCM)
    {
#ifdef TEST_MODE
        LOG_F(INFO, "Do IdleRun() for CM");
#endif //#ifdef TEST_MODE
        m_mbCM->IdleRun();
    }
    LOG_F(INFO, "InitObjectDetection performed from %s", (GetStringInitParams(*odInitParams)).c_str());
    // TODO: check if really OK
    return OD_ErrorCode::OD_OK;
}
OD_ErrorCode ObjectDetectionManagerHandler::PrepareOperateObjectDetection(OD_CycleInput *cycleInput)
{
#ifdef TEST_MODE
    cout << " PrepareOperateObjectDetection: Prepare to run on frame " << cycleInput->ImgID_input << endl;
#endif //#ifdef TEST_MODE

    if (cycleInput && cycleInput->ptr) // input is valid
    {

        OD_CycleInput *tempCycleInput = nullptr;

#ifdef TEST_MODE
        cout << "Replace old next cycle input" << endl;
#endif //#ifdef TEST_MODE

        m_mutexOnNext.lock();
        //glob_mutexOnNext.lock();
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
                LOG_F(WARNING, "PrepareOperateObjectDetection: attempt to call with same frame twice, skipping");
            }
        }
        else
        { // m_nextCycleInput is null, safe to create it
            m_nextCycleInput = NewCopyCycleInput(cycleInput, m_numPtrPixels);
        }
        m_mutexOnNext.unlock();
        //glob_mutexOnNext.unlock();
    }
    else
    {
        LOG_F(WARNING, "PrepareOperateObjectDetection:Input cycle is null, skip");
    }

#ifdef TEST_MODE
    if (m_nextCycleInput && m_nextCycleInput->ptr)
        cout << "PrepareOperateObjectDetection:Next cycle is valid" << endl;
    else
        cout << "PrepareOperateObjectDetection:Next cycle is empty" << endl;
#endif //#ifdef TEST_MODE

    //TODO: return not always OK

    return OD_ErrorCode::OD_OK;
}

void ObjectDetectionManagerHandler::IdleRun()
{
    cout << " ObjectDetectionManagerHandler::Idle Run (on neutral)" << endl;

    //TODO: is it continues in memory ?
    // unsigned char *tempPtr = new unsigned char[m_numImgPixels];
    std::vector<uint8_t> *img_data = new std::vector<uint8_t>(m_numImgPixels); //try
    //create temp ptr
    // this->m_mbATR->RunRGBVector(tempPtr, this->m_initParams->supportData.imageHeight, this->m_initParams->supportData.imageWidth);

    //try
    this->m_mbATR->RunRGBVector(*img_data, this->m_initParams->supportData.imageHeight, this->m_initParams->supportData.imageWidth);
    //delete tempPtr;
    delete img_data; //try
}

OD_ErrorCode ObjectDetectionManagerHandler::OperateObjectDetection(OD_CycleOutput *odOut)
{

#ifdef TEST_MODE
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
#endif //#ifdef TEST_MODE

    if (m_curCycleInput) // never suppose to happen, jic
    {
        LOG_F(ERROR, "This was never  supposed to happen... m_curCycleInput is not empty... How?");
        DeleteCycleInput(m_curCycleInput);
        m_curCycleInput = nullptr;
        return OD_ErrorCode::OD_FAILURE;
    }
    // deep copy next into current, next is not null here
    m_mutexOnNext.lock();
    // glob_mutexOnNext.lock();
    m_curCycleInput = SafeNewCopyCycleInput(m_nextCycleInput, m_numPtrPixels); // safe take care of NULL
    DeleteCycleInput(m_nextCycleInput);                                        //just to allow recursive call of OperateObjectDetection
    m_nextCycleInput = nullptr;
    m_mutexOnNext.unlock();
    //glob_mutexOnNext.unlock();

    if (!(m_curCycleInput && m_curCycleInput->ptr)) // former next (current ) cycle or buffer is null
    {
#ifdef TEST_MODE
        cout << "ObjectDetectionManagerHandler::OperateObjectDetection nothing todo" << endl;
        cout << "###UnLocked" << endl;
#endif //#ifdef TEST_MODE
        DeleteCycleInput(m_curCycleInput);
        m_curCycleInput = nullptr;

        return OD_ErrorCode::OD_OK;
    }

#ifdef TEST_MODE
    cout << " OperateObjectDetection: Run on frame " << m_curCycleInput->ImgID_input << endl;
#endif //#ifdef TEST_MODE

    unsigned int fi = m_curCycleInput->ImgID_input;
    int h = m_initParams->supportData.imageHeight;
    int w = m_initParams->supportData.imageWidth;

    e_OD_ColorImageType colortype = m_initParams->supportData.colorType;

    if (colortype == e_OD_ColorImageType::YUV422) // if raw
    {
        this->m_mbATR->RunRawImageFast(m_curCycleInput->ptr, h, w, (int)colortype);
    }
    else if (colortype == e_OD_ColorImageType::NV12) // if raw NV12
    {
        this->m_mbATR->RunRawImageFast(m_curCycleInput->ptr, h, w, (int)colortype);
    }
    else if (colortype == e_OD_ColorImageType::RGB) // if rgb
    {
#ifdef TEST_MODE
        cout << " Internal Run on RGB buffer " << endl;
#endif //#ifdef TEST_MODE
        this->m_mbATR->RunRGBVector(m_curCycleInput->ptr, h, w);
    }
    else if (colortype == e_OD_ColorImageType::RGB_IMG_PATH) //path
    {
#ifdef TEST_MODE
        cout << " Internal Run on RGB_IMG_PATH " << endl;
#endif //#ifdef TEST_MODE
        this->m_mbATR->RunRGBImgPath(m_curCycleInput->ptr);
    }
    else
    {
        return OD_ErrorCode::OD_ILEGAL_INPUT;
    }

#ifdef TEST_MODE
    cout << " ObjectDetectionManagerHandler::OperateObjectDetection starts PopulateCycleOutput  " << endl;
#endif //#ifdef TEST_MODE
    // save results
    this->PopulateCycleOutput(odOut);

    //Color Model (CM)
    if (odOut->numOfObjects > 0 && m_withActiveCM)
        m_mbCM->RunImgWithCycleOutput(m_mbATR->GetKeepImg(), odOut, 0, (odOut->numOfObjects - 1), true);

    odOut->ImgID_output = fi;

// copy current into prev
#ifdef TEST_MODE
    cout << "ObjectDetectionManagerHandler::OperateObjectDetection m_prevCycleInput <-- m_curCycleInput <-- nullptr" << endl;
#endif //#ifdef TEST_MODE

    OD_CycleInput *tempCI = m_prevCycleInput;
    m_prevCycleInput = m_curCycleInput; // transfere

    m_mutexOnPrev.lock();
    //glob_mutexOnPrev.lock();
    DeleteCycleInput(tempCI);
    m_mutexOnPrev.unlock();
    //glob_mutexOnPrev.unlock();

    m_curCycleInput = nullptr;

    // string outName = "outRes/out_res3_" + std::to_string(odOut->ImgID_output) + "_in.png";
    // this->SaveResultsATRimage(nullptr,odOut, (char*)outName.c_str(),false);

#ifdef TEST_MODE
    cout << "###UnLocked" << endl;
#endif //#ifdef TEST_MODE

    //Recoursive call
    if (m_nextCycleInput)
    {
#ifdef TEST_MODE
        cout << "+++++++++++++++++++++++++++++++++Call OperateObjectDetection again recoursively" << endl;
#endif //#ifdef TEST_MODE

        OperateObjectDetection(odOut);
    }
    return OD_ErrorCode::OD_OK;
}

bool ObjectDetectionManagerHandler::SaveResultsATRimage(OD_CycleOutput *co, char *imgNam, bool show)
{
    OD_CycleInput *tempci = nullptr;
    m_mutexOnPrev.lock();
    //glob_mutexOnPrev.lock();
    if (!(m_prevCycleInput && m_prevCycleInput->ptr))
    {
        LOG_F(WARNING, "ObjectDetectionManagerHandler::SaveResultsATRimage: No m_prevCycleInput data, skipping");
        return false;
    }
    else
    {
        // deep copy m_prevCycleInput
        tempci = NewCopyCycleInput(m_prevCycleInput, this->m_numPtrPixels);
    }
    m_mutexOnPrev.unlock();
    //glob_mutexOnPrev.unlock();

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
        myRGB = new cv::Mat(h, w, CV_8UC3);
        fastYUV2RGB((char *)(tempci->ptr), w, h, myRGB);
    }
    else if (colortype == e_OD_ColorImageType::NV12) // if NV12
    {
        myRGB = new cv::Mat(h, w, CV_8UC3);
        fastNV12ToRGB((char *)(tempci->ptr), w, h, myRGB);
    }
    else if (colortype == e_OD_ColorImageType::RGB || colortype == e_OD_ColorImageType::RGB_IMG_PATH) // if rgb
    {
        myRGB = new cv::Mat(h, w, CV_8UC3);
        std::copy(buffer, buffer + w * h * 3, myRGB->data);
#ifdef TEST_MODE
        cv::imwrite("debug_newImg1_bgr.tif", *myRGB);
#endif //#ifdef TEST_MODE
    }
    else
    {
        DeleteCycleInput(tempci);
        return false;
    }
#ifdef TEST_MODE
    std::cout << "***** num_detections " << co->numOfObjects << std::endl;
#endif //#ifdef TEST_MODE
    for (int i = 0; i < co->numOfObjects; i++)
    {
        int classId = co->ObjectsArr[i].tarClass;
        OD::e_OD_TargetColor colorId = co->ObjectsArr[i].tarColor;
        float score = co->ObjectsArr[i].tarScore;
        OD_BoundingBox bbox_data = co->ObjectsArr[i].tarBoundingBox;

        std::vector<float> bbox = {bbox_data.x1, bbox_data.x2, bbox_data.y1, bbox_data.y2};

        if (score >= drawThresh)
        {
#ifdef TEST_MODE
            cout << "add rectangle to drawing" << endl;
#endif //#ifdef TEST_MODE

            float x = bbox_data.x1;
            float y = bbox_data.y1;
            float right = bbox_data.x2;
            float bottom = bbox_data.y2;

            cv::rectangle(*myRGB, {(int)x, (int)y}, {(int)right, (int)bottom}, {125, 255, 51}, 2);
            cv::Scalar tColor(124, 200, 10);
            tColor = GetColor2Draw(colorId);

            cv::putText(*myRGB, string("Label:") + std::to_string(classId) + ";" + std::to_string(int(score * 100)) + "%", cv::Point(x, y - 10), 1, 2, tColor, 3);
        }
    }
#ifdef TEST_MODE
    cout << " Done reading targets" << endl;
#endif //#ifdef TEST_MODE
    if (show)
    {
        cv::Mat imgS;
        cv::resize(*myRGB, imgS, cv::Size(1365, 720));
        cv::imshow("Image", imgS);

        char c = (char)cv::waitKey(25);
    }
    cv::Mat bgr(h, w, CV_8UC3);
    cv::cvtColor(*myRGB, bgr, cv::COLOR_RGB2BGR);

#ifdef TEST_MODE
    cv::imwrite(imgNam, bgr);
    cout << " Done saving image" << endl;
#endif //#ifdef TEST_MODE

    if (myRGB != nullptr)
    {
        myRGB->release();
        delete myRGB;
    }
#ifdef TEST_MODE
    cout << " Done cleaning image" << endl;
#endif //#ifdef TEST_MODE
    DeleteCycleInput(tempci);
    return true;
}

int ObjectDetectionManagerHandler::PopulateCycleOutput(OD_CycleOutput *cycleOutput)
{
    float LOWER_SCORE_THRESHOLD = 0.1; //TODO: ini param (?)
#ifdef TEST_MODE
    cout << "ObjectDetectionManagerHandler::PopulateCycleOutput" << endl;
#endif //TEST_MODE

    OD_DetectionItem *odi = cycleOutput->ObjectsArr;

    int N = m_mbATR->GetResultNumDetections();

#ifdef TEST_MODE
    cout << "PopulateCycleOutput: Num detections total " << N << endl;
#endif //TEST_MODE

    auto bbox_data = m_mbATR->GetResultBoxes();
    unsigned int w = this->m_initParams->supportData.imageWidth;
    unsigned int h = this->m_initParams->supportData.imageHeight;

    cycleOutput->numOfObjects = N;
    for (int i = 0; i < N; i++)
    {
        e_OD_TargetClass aa = e_OD_TargetClass(1);
        odi[i].tarClass = e_OD_TargetClass(m_mbATR->GetResultClasses(i));
        odi[i].tarScore = m_mbATR->GetResultScores(i);
        if (odi[i].tarScore < LOWER_SCORE_THRESHOLD) // we suppose detections are sorted by score !!!
        {
            cycleOutput->numOfObjects = i;
#ifdef TEST_MODE
            cout << "Cutting PopulateCycleOutput. Taking only " << cycleOutput->numOfObjects << endl;
#endif //TEST_MODE
            break;
        }

        odi[i].tarBoundingBox = {bbox_data[i * 4 + 1] * w, bbox_data[i * 4 + 3] * w, bbox_data[i * 4] * h, bbox_data[i * 4 + 2] * h};
    }

    return cycleOutput->numOfObjects;
}

OD_ErrorCode ObjectDetectionManagerHandler::OperateObjectDetectionOnTiledSample(OD_CycleInput *cycleInput, OD_CycleOutput *cycleOutput)
{
    LOG_F(INFO, "Starting OperateObjectDetectionOnTiledSample");

    cycleOutput->numOfObjects = 0;

    uint bigH = m_initParams->supportData.imageHeight;
    uint bigW = m_initParams->supportData.imageWidth;

    const char *imgName = (const char *)cycleInput->ptr;

    //create tiled image
    cv::Mat *bigIm = new cv::Mat(bigH, bigW, CV_8UC3);
    bigIm->setTo(Scalar(0, 0, 0));

    std::list<float *> *tarList = new list<float *>(0);

    LOG_F(INFO, "Create tiled image from %s", imgName);
    CreateTiledImage(imgName, bigW, bigH, bigIm, tarList);

#ifdef TEST_MODE
    cv::imwrite("bigImg.tif", *bigIm);
#endif //#ifdef TEST_MODE

    unsigned char *ptrTif = ParseCvMat(*bigIm); // has new inside
    //run operate part without sync stuff etc.

#ifdef TEST_MODE
    cout << " Internal Run on RGB buffer " << endl;
#endif //#ifdef TEST_MODE

    this->m_mbATR->RunRGBVector(ptrTif, bigH, bigW);

    OD_CycleOutput *tempCycleOutput = NewOD_CycleOutput(350);
    this->PopulateCycleOutput(tempCycleOutput);

    // color
    if (m_withActiveCM && m_mbCM != nullptr && tempCycleOutput->numOfObjects > 0)
        this->m_mbCM->RunImgWithCycleOutput(*bigIm, tempCycleOutput, 0, tempCycleOutput->numOfObjects - 1, true);

//DEBUG
#ifdef TEST_MODE
    m_prevCycleInput = new OD_CycleInput(); // TODO: in ifdef (?) TODO take care of new
    m_prevCycleInput->ptr = ptrTif;
    SaveResultsATRimage(tempCycleOutput, (char *)"tiles1.png", false);
#endif //#ifdef TEST_MODE

    // analyze results and populate output
    AnalyzeTiledSample(tempCycleOutput, tarList, cycleOutput);

    // clean
    bigIm->release();
    delete bigIm;
    std::list<float *>::iterator it;
    for (it = tarList->begin(); it != tarList->end(); ++it)
        delete (*it);
    delete tarList;
    delete ptrTif;
    delete tempCycleOutput->ObjectsArr;
    delete tempCycleOutput;

    //TODO: take care of nothing detected
    return OD_ErrorCode::OD_OK;
}
int ObjectDetectionManagerHandler::CleanWrongTileDetections(OD_CycleOutput *co1, std::list<float *> *tarList)
{
    int numRemoved = 0;
    float thresh = 0.01;
    int numTrueTargets = tarList->size();
    float objBB[4];

    for (size_t d = 0; d < co1->numOfObjects; d++)
    {
        //object bb
        objBB[0] = co1->ObjectsArr[d].tarBoundingBox.x1;
        objBB[2] = co1->ObjectsArr[d].tarBoundingBox.x2;
        objBB[1] = co1->ObjectsArr[d].tarBoundingBox.y1;
        objBB[3] = co1->ObjectsArr[d].tarBoundingBox.y2;
        bool foundTarget = false;

        std::list<float *>::iterator it;
        for (it = tarList->begin(); it != tarList->end(); ++it)
        {
            //target bb
            float *targetBB = *it;
            float iou = IoU(targetBB, objBB);
            if (iou > thresh) //found target
            {
                foundTarget = 1;
                break;
            }
        }
        if (!foundTarget)
        { //TODO: remove object
            co1->ObjectsArr[d].tarScore = 0;
            numRemoved++;
        }
    }
    return numRemoved;
}
void ObjectDetectionManagerHandler::AnalyzeTiledSample(OD_CycleOutput *co1, std::list<float *> *tarList, OD_CycleOutput *co2)
{
    int MAX_TILES_CONSIDER = 3;

    // make sure co1->ObjectsArr[i] is one of tarList[j] by IoU
    int nr = CleanWrongTileDetections(co1, tarList);

#ifdef TEST_MODE
    cout << " CleanWrongTileDetections removed " << nr << " objects" << endl;
#endif //#ifdef TEST_MODE

    co1->numOfObjects = co1->numOfObjects - nr;
    int co1NumOfObjectsWithSkips = co1->numOfObjects + nr;

    for (size_t i = 0; i < co1NumOfObjectsWithSkips; i++)
    {
        if (co1->ObjectsArr[i].tarScore < 0.2)
            continue;

        // if already exists increment score
        int targetSlot = co2->numOfObjects;
        for (size_t i1 = 0; i1 < co2->numOfObjects; i1++)
        {
            if (co1->ObjectsArr[i].tarClass == co2->ObjectsArr[i1].tarClass)
                if (co1->ObjectsArr[i].tarSubClass == co2->ObjectsArr[i1].tarSubClass)
                    if (co1->ObjectsArr[i].tarColor == co2->ObjectsArr[i1].tarColor)
                    {
                        targetSlot = i1;
                        break;
                    }
        }

        // if not add element co2->numOfObjects, co2->numOfObjects++
        if (targetSlot == co2->numOfObjects)
        {
            if (co2->maxNumOfObjects <= co2->numOfObjects) //jic
                continue;
            co2->numOfObjects = co2->numOfObjects + 1;
            co2->ObjectsArr[targetSlot].tarScore = 0;
        }
        co2->ObjectsArr[targetSlot].tarClass = co1->ObjectsArr[i].tarClass;
        co2->ObjectsArr[targetSlot].tarSubClass = co1->ObjectsArr[i].tarSubClass;
        co2->ObjectsArr[targetSlot].tarColor = co1->ObjectsArr[i].tarColor;
        co2->ObjectsArr[targetSlot].tarScore += 1.0 / (co1->numOfObjects + 0.000001);
    }

    // sort co2->ObjectsArr[i2] by score
    bubbleSort_OD_DetectionItem(co2->ObjectsArr, co2->numOfObjects);

    // trim num objects
    if (co2->numOfObjects > MAX_TILES_CONSIDER)
        co2->numOfObjects = MAX_TILES_CONSIDER;

    //re-normalize score
    float totalScores = 0;
    for (size_t i2 = 0; i2 < co2->numOfObjects; i2++)
        totalScores = totalScores + co2->ObjectsArr[i2].tarScore;
    for (size_t i2 = 0; i2 < co2->numOfObjects; i2++)
        co2->ObjectsArr[i2].tarScore = co2->ObjectsArr[i2].tarScore / (totalScores + 0.00001);

    //TODO: compute scores based on Binomial distribution
}

bool ObjectDetectionManagerHandler::InitCM()
{
    LOG_F(INFO, "ObjectDetectionManagerHandler::InitCM");
    const char *modelPath;
    const char *ckpt;
    const char *inname;
    const char *outname;
    bool flag = false;
    std::string prepath = m_configParams->run_params["prePath"];
    //use  m_configParams
    for (size_t i = 0; i < m_configParams->models.size(); i++)
    {
#ifdef TEST_MODE
        std::cout << m_configParams->models[i]["nickname"] << std::endl;
        std::cout << m_configParams->models[i]["load_path"] << std::endl;
#endif //#ifdef TEST_MODE
        if (m_configParams->models[i]["nickname"].compare("default_CM") == 0)
        {
            modelPath = prepath.append(m_configParams->models[i]["load_path"]).c_str();
            inname = m_configParams->models[i]["input_layer"].c_str();
            outname = m_configParams->models[i]["output_layer"].c_str();
            if (m_configParams->models[i]["ckpt"].compare("nullptr") != 0)
                ckpt = m_configParams->models[i]["ckpt"].c_str();
            else
                ckpt = nullptr;
            flag = true;
            break;
        }
    }
    if (flag == false)
    {
        LOG_F(INFO, "default_CM is not specified,  `default` default CM loaded");
        modelPath = "graphs/output_graph.pb";
        ckpt = nullptr;
        inname = "conv2d_input";
        outname = "dense_1/Softmax";
    }

    //check file exist
    if (!file_exists_test(modelPath))
    {
        LOG_F(WARNING, "The color model file: %s  is missing... Skipping", modelPath);
        return false;
    }
    m_mbCM = new mbInterfaceCM();
    if (!m_mbCM->LoadNewModel(modelPath, ckpt, inname, outname))
    {
        LOG_F(ERROR, "Failed to load CM: %s\ninname: %s\noutname:%s", modelPath, inname, outname);
        return false;
    }
    LOG_F(INFO, "Loaded CM: %s\ninname: %s\noutname:%s", modelPath, inname, outname);
    return true;
}

void ObjectDetectionManagerHandler::SetConfigParams(InitParams *ip)
{
    m_configParams = ip;
}
InitParams *ObjectDetectionManagerHandler::GetConfigParams()
{
    return m_configParams;
}

bool ObjectDetectionManagerHandler::InitializeLogger()
{
    std::string lpath = m_configParams->run_params["logfile_path"];
    std::string prepath = m_configParams->run_params["prePath"];
    prepath.append(lpath);
    loguru::add_file(prepath.c_str(), loguru::Append, loguru::Verbosity_MAX);

    return true;
}

bool ObjectDetectionManagerHandler::InitConfigParamsFromFile(const char *iniFilePath)
{
    if (!file_exists_test(iniFilePath))
    {
        cout << "No file exists:" << iniFilePath << endl;
        return false;
    }
    m_configParams = new InitParams(iniFilePath);

    for (auto it = m_configParams->info.cbegin(); it != m_configParams->info.cend(); ++it)
    {
        std::cout << it->first << ": " << it->second << "\n";
    }

    std::cout << "Found a total of " << m_configParams->models.size() << " models" << std::endl;

    return true;
}