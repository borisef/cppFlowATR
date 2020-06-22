#include <iostream>
#include <fstream>
#include <chrono>

#include <cuda_runtime_api.h>

#include "cppflowCM/InterfaceCMTrt.h"
#include <utils/imgUtils.h>
#include <utils/odUtils.h>

//constructors/destructors
mbInterfaceCMTrt::mbInterfaceCMTrt() : mbInterfaceCMbase(128, 128, 7, 1, false),
                                       m_logger(TRTLoguruWrapper()), m_engine(nullptr), m_context(nullptr),
                                       m_DLACore(-1), m_isInitialized(false), m_bufferManager(nullptr)
{
#ifdef TEST_MODE
    std::cout << "Construct mbInterfaceCMTrt()" << std::endl;
#endif //#ifdef TEST_MODE
    initTrtResources();
}

mbInterfaceCMTrt::mbInterfaceCMTrt(int h, int w, int nc, int bs, bool hbs) : mbInterfaceCMbase(h, w, nc, bs, hbs),
                                                                             m_logger(TRTLoguruWrapper()), m_engine(nullptr), m_context(nullptr),
                                                                             m_DLACore(-1), m_isInitialized(false), m_bufferManager(nullptr)
{
#ifdef TEST_MODE
    cout << "Construct mbInterfaceCMTrt(h,w,...)" << endl;
#endif //#ifdef TEST_MODE
    initTrtResources();
}

mbInterfaceCMTrt::~mbInterfaceCMTrt()
{
#ifdef TEST_MODE
    cout << "Destruct mbInterfaceCMTrt" << endl;
#endif //#ifdef TEST_MODE
    if (m_bufferManager)
    {
        delete m_bufferManager;
    }
    if (m_context)
    {
        m_context->destroy();
    }
    if (m_engine)
    {
        m_engine->destroy();
    }
    cudaStreamDestroy(m_stream);
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_end);
}

void mbInterfaceCMTrt::initTrtResources()
{
    m_isInitialized = false;
    m_engine = nullptr;
    cudaSetDevice(0); //TODO: Consider taking cuda device from config
    m_DLACore = -1;   //TODO: Take m_DLACore from config
    initLibNvInferPlugins(&m_logger, "");
    cudaError_t ret;
    ret = cudaStreamCreate(&m_stream);
    if (ret != cudaSuccess)
    {
        LOG_F(ERROR, "FAILED to create cuda stream: %s", cudaGetErrorString(ret));
        return;
    }
    unsigned int cudaEventFlags = cudaEventDefault; // TODO: consider cudaEventBlockingSync, to allow calling cudaEventSynchronize(evt)
    ret = cudaEventCreateWithFlags(&m_start, cudaEventFlags);
    if (ret != cudaSuccess)
    {
        LOG_F(ERROR, "FAILED to create cuda event m_start: %s", cudaGetErrorString(ret));
        return;
    }
    ret = cudaEventCreateWithFlags(&m_end, cudaEventFlags);
    if (ret != cudaSuccess)
    {
        LOG_F(ERROR, "FAILED to create cuda event m_end: %s", cudaGetErrorString(ret));
        return;
    }
    LOG_F(INFO, "init succeed.");
    m_isInitialized = true;
}

bool mbInterfaceCMTrt::LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor)
{
#ifdef TEST_MODE
    std::cout << " LoadNewModel begin" << std::endl;
#endif //#ifdef TEST_MODE
    // deserializing an engine from a file
    if (!m_isInitialized)
    {
        LOG_F(ERROR, "LoadNewModel called before class initialization");
        return false;
    }
    if (m_context)
    {
        m_context->destroy();
    }
    std::ifstream engineFile = std::ifstream(modelPath, std::ios::binary);
    if (!engineFile)
    {
        LOG_F(ERROR, "Error opening engine file: %s", modelPath);
        return false;
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if (!engineFile)
    {
        LOG_F(ERROR, "Error reading engine file: %s", modelPath);
        return false;
    }

    auto runtime = nvinfer1::createInferRuntime(m_logger);
    if (m_DLACore != -1)
    {
        LOG_F(INFO, "Setting DLA core to: %s", m_DLACore);
        runtime->setDLACore(m_DLACore);
    }

    m_engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    runtime->destroy();
    if (m_engine)
    {
        LOG_F(INFO, "Engine file deserialized successfuly. CudaEngine at: %p", (void *)m_engine);
        // Create execution context
        m_context = m_engine->createExecutionContext();
        // TODO: Consider handling dynamic batch size in inference here.

        // Allocate host<->device buffers
        // Use an aliasing shared_ptr since we don't want engine to be deleted when bufferManager goes out of scope.
        std::shared_ptr<nvinfer1::ICudaEngine> emptyPtr{};
        std::shared_ptr<nvinfer1::ICudaEngine> aliasPtr(emptyPtr, m_engine);
        m_bufferManager = new BufferManager(aliasPtr, m_batchSize, m_context);
        //m_bufferManager = new BufferManager(aliasPtr, m_batchSize, m_batchSize > 1 ? nullptr : m_context);
        m_inTensors.clear();
        m_outTensors.clear();
        for (int i = 0; i < m_engine->getNbBindings(); i++)
        {
            std::map<int, std::string> &toInsertToIt(m_engine->bindingIsInput(i) ? m_inTensors : m_outTensors);
            toInsertToIt[i] = m_engine->getBindingName(i);
        }
    }
    LOG_F(INFO, "LoadNewModel is done. m_inTensors.size() = %d, m_outTensors.size() = %d.", m_inTensors.size(), m_outTensors.size());
    return (m_engine != nullptr &&
            // At the moment our color model requires exactly one input tensor and one output tensor
            m_inTensors.size() == 1 &&
            m_outTensors.size() == 1);
}

bool mbInterfaceCMTrt::doInference()
{
    // TODO: Adding profiling support here
    // // Dump inferencing time per layer basis
    // SimpleProfiler profiler("Layer time");
    // if (reporting.profile)
    // {
    //     context->setProfiler(&profiler);
    // }

    float totalGpu{0};  // GPU timer
    float totalHost{0}; // Host timer

    auto tStart = std::chrono::high_resolution_clock::now();
    cudaEventRecord(m_start, m_stream);


    // for batch inference
    // bool enqueue(int batchSize, void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed);
    // for one time inference
    // bool enqueueV2(void** bindings, cudaStream_t stream, cudaEvent_t* inputConsumed);
    // m_context->enqueueV2(&m_bufferManager->getDeviceBindings()[0], m_stream, nullptr);
    m_context->enqueue(m_batchSize, &m_bufferManager->getDeviceBindings()[0], m_stream, nullptr);
    //m_context->enqueue(m_batchSize, &m_bufferManager->getDeviceBindings()[0], stream, nullptr);

    cudaEventRecord(m_end, m_stream);
    cudaEventSynchronize(m_end);

    auto tEnd = std::chrono::high_resolution_clock::now();
    totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    cudaEventElapsedTime(&totalGpu, m_start, m_end);

    // if (reporting.profile)
    // {
    //     gLogInfo << profiler;
    // }

    return true;
}

bool mbInterfaceCMTrt::normalizeImageIntoInputBuffer(const cv::Mat &img)
{
    cv::Mat img_resized;
    size_t img_resized_data_size(m_patchWidth * m_patchHeight * 3);
    cv::resize(img, img_resized, cv::Size(m_patchWidth, m_patchHeight));
    float *hostInputDataBuffer = reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_inTensors.begin()->second));
    if (hostInputDataBuffer == nullptr)
    {
        return false; //failed to get hostInputDataBuffer
    }
    for (size_t i = 0; i < img_resized_data_size; i++)
    {
        hostInputDataBuffer[i] = img_resized.data[i] / 255.0;
    }
    return (0 == m_bufferManager->copyInputToDevice());
}

bool mbInterfaceCMTrt::normalizeImagesIntoInputBuffer(const std::vector<cv::Mat> &images)
{
    cv::Mat img_resized;
    size_t img_resized_data_size(m_patchWidth * m_patchHeight * 3);
    float *hostInputDataBuffer = reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_inTensors.begin()->second));
    if (hostInputDataBuffer == nullptr)
    {
        return false; //failed to get hostInputDataBuffer
    }
    size_t j = 0;
    for (const cv::Mat &img : images)
    {
        cv::resize(img, img_resized, cv::Size(m_patchWidth, m_patchHeight));
        for (size_t i = 0; i < img_resized_data_size; i++, j++)
        {
            hostInputDataBuffer[j] = img_resized.data[i] / 255.0;
        }
    }
    return (0 == m_bufferManager->copyInputToDevice());
}

std::vector<float> mbInterfaceCMTrt::RunRGBimage(cv::Mat img)
{
    std::vector<float> ans;
    if (!normalizeImageIntoInputBuffer(img))
    {
        return ans;
    }

    doInference();

    if (0 != m_bufferManager->copyOutputToHost())
    {
        return ans;
    }

    float *out_data = reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_outTensors.begin()->second));
    ans = std::vector<float>(out_data, out_data + m_numColors);
    return ans;
}

std::vector<std::vector<float>> mbInterfaceCMTrt::RunRGBimagesBatch(const std::vector<cv::Mat> &images)
{
    std::vector<std::vector<float>> ans;
    if (!normalizeImagesIntoInputBuffer(images))
    {
        return ans;
    }

    doInference();

    if (0 != m_bufferManager->copyOutputToHost())
    {
        return ans;
    }

    float *out_data = reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_outTensors.begin()->second));
    for (size_t i = 0 ; i < images.size(); i++)
    {
        int begin_offset = i * m_numColors;
        int end_offset = begin_offset + m_numColors;
        std::vector<float> one_image_res = std::vector<float>(out_data + begin_offset, out_data + end_offset);
        ans.push_back(one_image_res);
    }
    return ans;
}

std::vector<float> mbInterfaceCMTrt::RunRGBImgPath(const unsigned char *ptr)
{
std:
    string filename((const char *)ptr);
    cv::Mat img = cv::imread(filename);
    return RunRGBimage(img);
}

void mbInterfaceCMTrt::IdleRun()
{
    if (isInitialized())
    {
        doInference();
    }
}

std::vector<float> mbInterfaceCMTrt::RunImgBB(cv::Mat img, OD::OD_BoundingBox bb)
{
    cv::Mat croppedRef, img_resized;
#ifdef TEST_MODE
    cv::Mat debugImg = img.clone();
#endif //#ifdef TEST_MODE

    //get sub-image
    //TODO: take tileMargin into account
    cv::Rect myROI(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);

    croppedRef = img(myROI);

#ifdef TEST_MODE
    cv::rectangle(debugImg, myROI, cv::Scalar(0, 255, 0), 5);
    cv::imwrite("t1.png", img);
    cv::imwrite("t2.png", croppedRef);
#endif //#ifdef TEST_MODE

    //resize
    cv::resize(croppedRef, img_resized, cv::Size(m_patchWidth, m_patchHeight));

#ifdef TEST_MODE
    cv::imwrite("t1a.png", debugImg);
    cv::imwrite("t3.png", img_resized);
#endif //#ifdef TEST_MODE

    //apply
    return RunRGBimage(croppedRef);
}

bool mbInterfaceCMTrt::RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults)
{
#ifdef TEST_MODE
    cv::Mat debugImg = img.clone();
#endif                        //#ifdef TEST_MODE
    int N = co->numOfObjects; // N can be smaller or bigger than BS
    int origStopInd = stopInd;

    int BS = stopInd - startInd + 1; // requested batch size
    if (m_hardBatchSize)
        BS = m_batchSize; // force BS
    //BS = 1; //TODO: Support batches
    //std::vector<float> inVec(BS * m_patchHeight * m_patchWidth * 3);
    int tempStopInd = stopInd;
    while (1)
    {
        if (tempStopInd - startInd + 1 > BS)
        {
            tempStopInd = BS - 1 + startInd;
        }

        // prepare batch input fo inference:
        std::vector<cv::Mat> batchIn;
        for (size_t i = startInd; i <= tempStopInd; i++)
        {
            if (i >= N)
            {
                //jic, not suppose to happen
                break;
            }
            cv::Mat croppedRef;
            //crop
            OD::OD_BoundingBox bb = co->ObjectsArr[i].tarBoundingBox;
            //TODO: take tileMargin into account
            float dw = bb.x2 - bb.x1;
            float dh = bb.y2 - bb.y1;
            float x1 = bb.x1 - dw * m_tileMargin;
            float y1 = bb.y1 - dh * m_tileMargin;
            float x2 = x1 + dw * (1.0f + 2.0f * m_tileMargin);
            float y2 = y1 + dh * (1.0f + 2.0f * m_tileMargin);
            cv::Rect myROI;
            if (x1 > 0 && y1 > 0 && y2 < img.rows && x2 < img.cols)
                myROI = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            else
                myROI = cv::Rect(bb.x1, bb.y1, bb.x2 - bb.x1, bb.y2 - bb.y1);

#ifdef TEST_MODE
            cv::rectangle(debugImg, myROI, cv::Scalar(0, 255, 0), 5);
#endif //#ifdef TEST_MODE
            croppedRef = img(myROI);
            //prepare the batch:
            batchIn.push_back(croppedRef);

#ifdef TEST_MODE
            cv::imwrite("color_batch.png", debugImg);
#endif //#ifdef TEST_MODE
        }
        std::vector<std::vector<float>> batchResults = RunRGBimagesBatch(batchIn);
        if (copyResults)
        {
            for (size_t si = startInd; si <= tempStopInd; si++)
            {
                //subvector
                vector<float> outRes = batchResults[si - startInd];
                //get color
                //argmax
                uint color_id = std::distance(outRes.begin(), std::max_element(outRes.begin(), outRes.end()));

#ifdef TEST_MODE
                cout << "color id = " << color_id << endl;
                static const char *cid_to_cname[] = {"black",   // 0
                                                     "blue",    // 1
                                                     "gray",    // 2
                                                     "green",   // 3
                                                     "red",     // 4
                                                     "white",   // 5
                                                     "yellow"}; // 6
                const char *color_name = "UNKNOWN_COLOR";
                if (color_id < 7)
                {
                    color_name = cid_to_cname[color_id];
                }
                cout << "Color: " << color_name << endl;
                //PrintColor(color_id);
                // score
                cout << "Net score: " << outRes[color_id] << endl;
#endif //#ifdef TEST_MODE
    // copy res into co
                co->ObjectsArr[si].tarColor = TargetColor(color_id);
                co->ObjectsArr[si].tarColorScore = outRes[color_id];
            }
        }
        if (tempStopInd >= origStopInd)
        {
            break;
        }
        else
        {
            startInd = tempStopInd + 1;
            tempStopInd = origStopInd;
        }
    }
    return true;
}
