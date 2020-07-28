#include <cppflowATR/InterfaceATR.h>
#include <utils/imgUtils.h>
#include <utils/odUtils.h>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;

#ifndef NO_TRT
std::string dimsToStr(const nvinfer1::Dims &dims);
#endif

void mbInterfaceATR::InitTRTStuff()
{
#ifndef NO_TRT
    // Init tensor-rt related resources:
    m_engine = nullptr;
    m_context = nullptr;
    m_isInitialized = false;
    m_bufferManager = nullptr;
    m_isInitialized = false;
    m_engine = nullptr;
    m_batchSizeTRT = 1; // at the moment only 1 image batch is suported
    cudaSetDevice(0);   // TODO: Consider taking cuda device from config
    m_DLACore = -1;     // TODO: Take m_DLACore from config
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
#endif // #ifndef NO_TRT
}

mbInterfaceATR::mbInterfaceATR()
{
#ifdef TEST_MODE
    cout << "Construct mbInterfaceATR" << endl;
#endif //TEST_MODE
    m_model = nullptr;
    m_outTensorNumDetections = nullptr;
    m_outTensorScores = nullptr;
    m_outTensorBB = nullptr;
    m_outTensorClasses = nullptr;
    m_inpName = nullptr;
    m_modelType = e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_UNKNOWN;
    InitTRTStuff();
}

void mbInterfaceATR::DestroyTRTStuff()
{
#ifndef NO_TRT
    if (m_bufferManager)
    {
        delete m_bufferManager;
        m_bufferManager = nullptr;
    }
    if (m_context)
    {
        m_context->destroy();
        m_context = nullptr;
    }
    if (m_engine)
    {
        m_engine->destroy();
        m_engine = nullptr;
    }
    cudaStreamDestroy(m_stream);
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_end);
    m_isInitialized = false;
#endif //#ifndef NO_TRT
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
    DestroyTRTStuff();
}

bool mbInterfaceATR::LoadNewTRTModel(const char *modelPath)
{
#ifdef NO_TRT
    LOG_F(ERROR, "LoadNewTRTModel() called but TensorRT is not supported.");
    return false;
#else
    if (!m_isInitialized)
    {
        LOG_F(ERROR, "LoadNewTRTModel called before class initialization");
        return false;
    }
    if (m_bufferManager)
    {
        delete m_bufferManager;
        m_bufferManager = nullptr;
    }
    if (m_context)
    {
        m_context->destroy();
        m_context = nullptr;
    }
    if (m_engine)
    {
        m_engine->destroy();
        m_engine = nullptr;
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
        m_bufferManager = new BufferManager(aliasPtr, m_batchSizeTRT, m_context);
        //m_bufferManager = new BufferManager(aliasPtr, m_batchSize, m_batchSize > 1 ? nullptr : m_context);
        m_inTensors.clear();
        m_outTensors.clear();
        for (int i = 0; i < m_engine->getNbBindings(); i++)
        {
            const char *tensor_name = m_engine->getBindingName(i);
            const nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
            if (m_engine->bindingIsInput(i))
            {
                m_inTensors[i] = tensor_name;
                m_inputDims.m_numChannels = dims.d[0];
                m_inputDims.m_height = dims.d[1];
                m_inputDims.m_width = dims.d[2];
            }
            else
            {
                m_outTensors[i] = m_engine->getBindingName(i);
                if (m_outTensors[i] == "dense_class_td/Softmax") //TODO: get output tensor name from config
                {
                    m_numProposals = m_engine->getBindingDimensions(i).d[0];
                    m_numClasses = m_engine->getBindingDimensions(i).d[1]; // this includes the background class
                    LOG_F(INFO, "This modal proposes %d ROI's and classifies each into one of %d classes (including background)",
                          m_numProposals, m_numClasses);
                    m_trtAtrDetections.reserve(m_numProposals);
                    m_regressBBCoords.reserve(m_numProposals * 4 * (m_numClasses - 1));
                }
            }
            LOG_F(INFO, "Tensor[%d] is an %s tensor, named: %s, dims: %s",
                  i, (m_engine->bindingIsInput(i) ? "input" : "output"), tensor_name, dimsToStr(dims).c_str());
        }
    }
    LOG_F(INFO, "LoadNewTRTModel is done. m_inTensors.size() = %d, m_outTensors.size() = %d.", m_inTensors.size(), m_outTensors.size());
    return (m_engine != nullptr && m_inTensors.size() >= 1 && m_outTensors.size() >= 1);
#endif //#ifndef NO_TRT
}

std::string getFileExtention(const std::string &s)
{
    size_t dot_pos = s.rfind('.', s.length());
    std::string ext = "";
    return dot_pos != string::npos ? s.substr(dot_pos + 1) : "";
}

bool mbInterfaceATR::LoadNewModel(const char *modelPath)
{
    LOG_F(INFO, "LoadNewModel begin. file: %s", modelPath);
#ifdef TEST_MODE
    std::cout << " LoadNewModel begin. file: " << modelPath << std::endl;
#endif //TEST_MODE
    //delete previous model
    if (m_model != nullptr)
    {
        delete m_model;
        delete m_outTensorNumDetections;
        delete m_outTensorScores;
        delete m_outTensorBB;
        delete m_outTensorClasses;
        delete m_inpName;
    }
    m_modelType = e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_UNKNOWN;
    std::string model_type = getFileExtention(modelPath);
#ifdef TEST_MODE
    std::cout << " Parsed model type is: " << model_type << ", model file was: " << modelPath << std::endl;
#endif //TEST_MODE

    if (model_type == "pb")
    {
        m_modelType = e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TF;
        m_model = new Model(modelPath, CreateSessionOptions(0.3));
        m_outTensorNumDetections = new Tensor(*m_model, "num_detections");
        m_outTensorScores = new Tensor(*m_model, "detection_scores");
        m_outTensorBB = new Tensor(*m_model, "detection_boxes");
        m_outTensorClasses = new Tensor(*m_model, "detection_classes");
        m_inpName = new Tensor(*m_model, "image_tensor");
    }
    else if (model_type == "engine")
    {
#ifdef NO_TRT
        LOG_F(ERROR, "This is a NON-Tensor-RT Version. Cannot load engine file: %s", modelPath);
        return false;
#else
        m_modelType = e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT;
        return LoadNewTRTModel(modelPath);
#endif
    }
    else
    {
        m_modelType = e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_UNKNOWN;
        LOG_F(ERROR, "Cannot load model. Unsupported file extension (%s), parsed from file: %s", model_type.c_str(), modelPath);
        return false;
    }

    return true;
}

bool mbInterfaceATR::getTRTOutput()
{
#ifdef NO_TRT
    LOG_F(ERROR, "TensorRT is not supported in this version!!!");
    return false;
#else
    m_regressBBCoords.clear();
    m_trtAtrDetections.clear();
    if (0 != m_bufferManager->copyOutputToHost())
    {
        return false;
    }
    //TODO: get output tensor names from config
    float *proposals = reinterpret_cast<float *>(m_bufferManager->getHostBuffer("proposal"));                        // Dims: [m_numProposals, m_numClasses]
    float *per_class_scores = reinterpret_cast<float *>(m_bufferManager->getHostBuffer("dense_class_td/Softmax"));   // Dims: [m_numProposals, m_numClasses]
    float *fg_regress_boxes = reinterpret_cast<float *>(m_bufferManager->getHostBuffer("dense_regress_td/BiasAdd")); // Dims: [m_numProposals, 4 * (m_numClasses - 1)]

    // find num of non-background detections:
    int bgIndex = m_numClasses - 1;
    for (int i = 0; i < m_numProposals; i++)
    {
        float *scores_begin = &per_class_scores[i * m_numClasses];
        float *max_e = std::max_element(scores_begin, scores_begin + m_numClasses);
        int class_index = max_e - scores_begin;
        if (class_index != bgIndex) //class_index
        {
            //TODO: Make sure that threshold by class is not required here.
            float *bboxes_begin = fg_regress_boxes + (i * 4 * m_numClasses) + 4 * class_index;
            float y1 = proposals[i * 4 + 0] * m_inputDims.m_height;
            float x1 = proposals[i * 4 + 1] * m_inputDims.m_width;
            float y2 = proposals[i * 4 + 2] * m_inputDims.m_height;
            float x2 = proposals[i * 4 + 3] * m_inputDims.m_width;
            float x = x1, y = y1, w = x2 - x1, h = y2 - y1;
            float tx = bboxes_begin[0], ty = bboxes_begin[1], tw = bboxes_begin[2], th = bboxes_begin[3];

            tx /= 10.0; // these consts are from spec.rcnn_regr_std
            ty /= 10.0;
            tw /= 5.0;
            th /= 5.0;

            //def apply_regr(x, y, w, h, tx, ty, tw, th):
            float cx = x + w / 2.0;
            float cy = y + h / 2.0;
            float cx1 = tx * w + cx;
            float cy1 = ty * h + cy;
            tw = tw > 50 ? 50 : tw;
            th = th > 50 ? 50 : th;
            float w1 = std::exp(tw) * w;
            float h1 = std::exp(th) * h;
            x1 = cx1 - w1 / 2.0;
            y1 = cy1 - h1 / 2.0;
            x = x1;
            y = y1;
            x1 = x + w1;
            y1 = y + h1;
            x = x < 0 ? 0 : x;
            y = y < 0 ? 0 : y;
            x1 = x1 < 0 ? 0 : x1;
            y1 = y1 < 0 ? 0 : y1;
            x = x / m_inputDims.m_width;
            y = y / m_inputDims.m_height;
            x1 = x1 / m_inputDims.m_width;
            y1 = y1 / m_inputDims.m_height;
            float bb[4] = {y, x, y1, x1};
            //avoid zero-sized boxes (reuqire at 2x2 pixels size):
            if (x1 - x < 0 || y1 - y < 0)
            {
                LOG_F(INFO, "DBG: Filtered out too small box: ((x1, y1), (x2, y2)) = ((%f, %f), (%f, %f))", x, y, x1, y1);
                continue;
            }

            // class mapping
            // tlt's outputs are lexicographically sorted
            // ['car', 'bus', 'truck', 'van', 'jeep', 'pickup', 'pickup_open', 'pickup_close', 'forklift', 'tractor', 'station'

            static ATR_TargetSubClass_MB trt_class_to_ATR_TargetSubClass_MB[] = {
                ATR_TargetSubClass_MB::ATR_BUS,
                ATR_TargetSubClass_MB::ATR_CAR,
                ATR_TargetSubClass_MB::ATR_FORKLIFT,
                ATR_TargetSubClass_MB::ATR_JEEP,
                ATR_TargetSubClass_MB::ATR_PICKUP_CLOSED,
                ATR_TargetSubClass_MB::ATR_PICKUP_OPEN,
                ATR_TargetSubClass_MB::ATR_STATION,
                ATR_TargetSubClass_MB::ATR_TRACKTOR,
                ATR_TargetSubClass_MB::ATR_TRUCK,
                ATR_TargetSubClass_MB::ATR_VAN,
                ATR_TargetSubClass_MB::ATR_OTHER,
            };

#ifdef TEST_MODE
            std::cout << "bb: [" << x << ", " << y << ", " << x1 << ", " << y1 << "], class: " << class_index << ", score: " << *max_e << std::endl;
#endif //TEST_MODE
            m_trtAtrDetections.push_back(trtAtrDetection(bb, trt_class_to_ATR_TargetSubClass_MB[class_index], *max_e));
        }
    }
    LOG_F(INFO, "%d objects found on this inference.", m_trtAtrDetections.size());
    std::sort(m_trtAtrDetections.begin(), m_trtAtrDetections.end(), std::greater<trtAtrDetection>());
    //copy back the score-wise sotred bounding-boxes vector:
    for (const trtAtrDetection &det : m_trtAtrDetections)
    {
        m_regressBBCoords.insert(m_regressBBCoords.end(), det.bb_coords, det.bb_coords + 4);
    }
    return true;
#endif //#ifndef NO_TRT
}

int mbInterfaceATR::RunRGBimage(cv::Mat inp)
{
    // Put image in vector
    std::vector<uint8_t> img_data(inp.data, inp.data + inp.total() * inp.channels());
    RunRGBVector(img_data, inp.rows, inp.cols);
    //m_inpName->set_data(img_data, {1, inp.rows, inp.cols, inp.channels()});
    //m_model->run(m_inpName, {m_outTensorNumDetections, m_outTensorScores, m_outTensorBB, m_outTensorClasses});

    inp.copyTo(m_keepImg);

#ifdef TEST_MODE
    cv::imwrite("m_keepImg.png", m_keepImg);
#endif //TEST_MODE

    return 1;
}
int mbInterfaceATR::RunRGBImgPath(const unsigned char *ptr, float resize_factor)
{
#ifdef OPENCV_MAJOR_4
    cv::Mat inp1 = cv::imread(string((const char *)ptr), IMREAD_COLOR); //CV_LOAD_IMAGE_COLOR
    cv::cvtColor(inp1, inp1, cv::COLOR_BGR2RGB);                        //CV_BGR2RGB, 4
#else
    cv::Mat inp1 = cv::imread(string((const char *)ptr), CV_LOAD_IMAGE_COLOR);                                                   //
    cv::cvtColor(inp1, inp1, CV_BGR2RGB);                                                                                        //, 4
#endif
    if (resize_factor > 0 && resize_factor != 1)
    {
        //imresize of inp1 inplace
#ifdef OPENCV_MAJOR_4
        cv::resize(inp1, inp1, cv::Size(int(inp1.cols * resize_factor), int(inp1.rows * resize_factor)), 0, 0, INTER_LINEAR); //CV_INTER_LINEAR
#else
        cv::resize(inp1, inp1, cv::Size(int(inp1.cols * resize_factor), int(inp1.rows * resize_factor)), 0, 0, CV_INTER_LINEAR); //
#endif
    }

    return RunRGBimage(inp1);
}

int mbInterfaceATR::RunRGBVector(const unsigned char *ptr, int height, int width, float resize_factor)
{

#ifdef TEST_MODE
    cout << " RunRGBVector:Internal Run on RGB Vector on ptr*" << endl;
    cout << "RunRGBVector " << height << " " << width << "prt[10]" << ptr[10] << endl;
#endif //TEST_MODE

    unsigned char *buffer = (unsigned char *)ptr;

#ifdef TEST_MODE
    cout << " RunRGBVector:casted buffer to unsigned char* " << endl;
#endif //TEST_MODE

    cv::Mat tempIm(height, width, CV_8UC3, (void *)ptr);
#ifdef TEST_MODE
    cout << " RunRGBVector:copy buffer to cv::Mat* " << endl;
#endif //TEST_MODE
    //TODO: tempIm.data = (unsigned char *)ptr;

#ifdef TEST_MODE
    cv::imwrite("tempim.png", tempIm);
#endif //TEST_MODE

    //tempIm.copyTo(m_keepImg);//TODO: clone instead ?
    m_keepImg = tempIm.clone();
    cv::cvtColor(m_keepImg, m_keepImg, cv::COLOR_RGB2BGR);
#ifdef TEST_MODE
    cv::imwrite("m_keepImg.png", m_keepImg);
#endif //TEST_MODE

    cv::Size targetSize(int(tempIm.cols * resize_factor), int(tempIm.rows * resize_factor));
#ifndef NO_TRT
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
        // with tensor-rt we require input image to be resized to fit the network input
        targetSize = cv::Size(m_inputDims.m_width, m_inputDims.m_height);
    }
#endif
    if (tempIm.size() != targetSize)
    {
        LOG_F(INFO, "Reshaping input from (%d, %d) to (%d, %d)", tempIm.cols, tempIm.rows, targetSize.width, targetSize.height);
        cv::resize(tempIm, tempIm, targetSize, 0, 0, INTER_LINEAR); //CV_INTER_LINEAR
    }
#ifdef TEST_MODE
    cv::imwrite("tempim_resized.png", tempIm);
#endif                                         //TEST_MODE
    buffer = (unsigned char *)tempIm.data; //suppose it is continues

#ifdef TEST_MODE
    cout << " RunRGBVector:saving cv::Mat* " << endl;
    cv::imwrite("testRGBbuffer.tif", tempIm);
#endif //TEST_MODE

    std::vector<uint8_t> img_data(buffer, buffer + tempIm.rows * tempIm.cols * tempIm.channels());

    return (RunRGBVector(img_data, int(height * resize_factor), int(width * resize_factor)));
}
int mbInterfaceATR::RunRGBVector(std::vector<uint8_t> &img_data, int height, int width, float resize_factor)
{
#ifdef TEST_MODE
    cout << " RunRGBVector:Internal Run on RGB Vector on vector<uint8_t> " << endl;
#endif //TEST_MODE
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TF)
    {
        // Put image in Tensor
        m_inpName->set_data(img_data, {1, height, width, 3});
        m_model->run(m_inpName, {m_outTensorNumDetections, m_outTensorScores, m_outTensorBB, m_outTensorClasses});
    }
    else if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
#ifdef NO_TRT
        LOG_F(ERROR, "Model type is TensorRT but it is not supported in this version!!!");
        return -1;
#else
        float totalHost{0};
        auto tStart = std::chrono::high_resolution_clock::now();
        if (!imageToTRTInputBuffer(img_data))
        {
            LOG_F(ERROR, "Failed to copy input image ");
            return -1;
        }
        auto tEnd = std::chrono::high_resolution_clock::now();
        totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        LOG_F(INFO, "last imageToTRTInputBuffer took: %f millis", totalHost);
        tStart = std::chrono::high_resolution_clock::now();
        doTRTInference();
        tEnd = std::chrono::high_resolution_clock::now();
        totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        LOG_F(INFO, "last doTRTInference took: %f millis", totalHost);
        tStart = std::chrono::high_resolution_clock::now();
        if (getTRTOutput() == false) //something went wrong
        {
            return -1;
        }
        tEnd = std::chrono::high_resolution_clock::now();
        totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        LOG_F(INFO, "last getTRTOutput took: %f millis", totalHost);
#endif //#ifndef NO_TRT
    }
    return 1; //TODO useful return
}

bool mbInterfaceATR::doTRTInference()
{
#ifdef NO_TRT
    LOG_F(ERROR, "TensorRT is not supported in this version!!!");
    return false;
#else
    // TODO: Adding profiling support here
    // // Dump inferencing time per layer basis
    // SimpleProfiler profiler("Layer time");
    // if (reporting.profile)
    // {
    //     context->setProfiler(&profiler);
    // }

    //float totalGpu{0};  // GPU timer
    //float totalHost{0}; // Host timer

    //auto tStart = std::chrono::high_resolution_clock::now();
    cudaEventRecord(m_start, m_stream);

    m_context->enqueue(m_batchSizeTRT, &m_bufferManager->getDeviceBindings()[0], m_stream, nullptr);

    cudaEventRecord(m_end, m_stream);
    cudaEventSynchronize(m_end);

    //auto tEnd = std::chrono::high_resolution_clock::now();
    //totalHost += std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    //cudaEventElapsedTime(&totalGpu, m_start, m_end);

    // if (reporting.profile)
    // {
    //     gLogInfo << profiler;
    // }

    return true;
#endif //#ifndef NO_TRT
}

#ifndef NO_TRT
std::string dimsToStr(const nvinfer1::Dims &dims)
{
    //dims str:
    stringstream sstr;
    int i = 0;
    for (; i < dims.nbDims - 1 && i < dims.MAX_DIMS; i++)
    {
        sstr << dims.d[i] << ", ";
    }
    if (i < dims.nbDims)
    {
        sstr << dims.d[i];
    }
    return sstr.str();
}
#endif //#ifndef NO_TRT

bool mbInterfaceATR::imageToTRTInputBuffer(const std::vector<uint8_t> &img_data)
{
#ifndef NO_TRT
    float *hostInputDataBuffer = reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_inTensors.begin()->second));
    // validity check of input size
    size_t expected_in_sz = m_bufferManager->size(m_inTensors.begin()->second) / sizeof(float);
    nvinfer1::Dims dims = m_engine->getBindingDimensions(m_inTensors.begin()->first);
    if (expected_in_sz != img_data.size())
    {
        std::string dims_str = dimsToStr(dims);
        LOG_F(WARNING, "Input size mismach. Got: %d elements vector for expected network input of %d (%s) elements",
              img_data.size(), expected_in_sz, dims_str.c_str());
    }
    else
    {
        //HWC: offset = h * im.rows * im.elemSize() + w * im.elemSize() + c
        //CHW: offset = c * im.rows * im.cols + h * im.cols + w
        float rgb_channel_mean[] = {123.68, 116.779, 103.939};
        const int C = dims.d[0], H = dims.d[1], W = dims.d[2];
        double max_elem = *std::max_element(img_data.begin(),img_data.end());
        double min_elem = *std::min_element(img_data.begin(),img_data.end());

        std::cout<<"C,H,W : "<< C<<','<<H<<','<<W<<", max value is: "<<max_elem<<", min value is: "<<min_elem<<std::endl;
        std::cout<<"img_data size: "<<img_data.size()<<std::endl;
        for (int ch = C-1; ch >= 0; ch--) //RGB->BGR
        //for (int ch = 0; ch < C; ch++)
        {
            float channel_mean = rgb_channel_mean[ch];
            for (int row = 0; row < H; row++)
            {
                for (int col = 0; col < W; col++)
                {
                    *hostInputDataBuffer++ = img_data[row * W * C + col * C + ch];
                }
            }
        }

    cv::Mat imm(3040,4056,CV_32FC3,reinterpret_cast<float *>(m_bufferManager->getHostBuffer(m_inTensors.begin()->second)));
    //cv::Mat imm(3040,4056,CV_8UC3,(float *)img_data.data());

    std::cout<<imm.size()<<std::endl;
    //cout <<"data:"<<imm.rowRange(0, 2)<<std::endl;
    cv::Mat imm2;
    //imm.convertTo(imm2,CV_8UC3);
    cv::imwrite("blabla.jpg",imm);

    }

    return (0 == m_bufferManager->copyInputToDevice());
#endif //#ifndef NO_TRT
    return false;
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
#ifdef TEST_MODE
    cv::imwrite("m_keepImg.png", m_keepImg);
#endif            //TEST_MODE
    delete myRGB; //??? TODO: is it safe?
    int status = RunRGBVector(img_data, height, width);

    return status;
}

int mbInterfaceATR::RunRawImageFast(const unsigned char *ptr, int height, int width, int colorType, float resize_factor)
{

    std::vector<uint8_t> img_data(int(height * resize_factor) * int(width * resize_factor) * 2);
    unsigned char *buffer = (unsigned char *)ptr;

    cv::Mat *myRGB = new cv::Mat(height, width, CV_8UC3);

    if (colorType == 7) //NV12
        fastNV12ToRGB((char *)ptr, width, height, myRGB);
    else //YUV422
        fastYUV2RGB((char *)ptr, width, height, myRGB);

#ifdef TEST_MODE
    // save JPG for debug
    cv::imwrite("debug_raw2rgb.tif", *myRGB);
#endif //TEST_MODE \
       //TODO: BGR -> RGB

    myRGB->copyTo(m_keepImg);
    cv::cvtColor(m_keepImg, m_keepImg, cv::COLOR_BGR2RGB); //???

#ifdef TEST_MODE
    cv::imwrite("m_keepImg.png", m_keepImg);
#endif //TEST_MODE

    if (resize_factor > 0 && resize_factor != 1)
    {
//imresize of myRGB inplace
#ifdef OPENCV_MAJOR_4
        cv::resize(*myRGB, *myRGB, cv::Size(int(myRGB->cols * resize_factor), int(myRGB->rows * resize_factor)), 0, 0, INTER_LINEAR); //CV_INTER_LINEAR
#else
        cv::resize(*myRGB, *myRGB, cv::Size(int(myRGB->cols * resize_factor), int(myRGB->rows * resize_factor)), 0, 0, CV_INTER_LINEAR);
#endif

#ifdef TEST_MODE
        // save JPG for debug
        cv::imwrite("debug_raw2rgb_resized.tif", *myRGB);
#endif //TEST_MODE
    }
    img_data.assign(myRGB->data, myRGB->data + myRGB->total() * myRGB->channels());

    delete myRGB; //??? TODO: is it safe?
    int status = RunRGBVector(img_data, int(height * resize_factor), int(width * resize_factor));

    return status;
}

int mbInterfaceATR::GetResultNumDetections()
{
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
#ifdef NO_TRT
        LOG_F(ERROR, "Model type is TensorRT but it is not supported in this version!!!");
        return 0;
#else
        return m_trtAtrDetections.size();
#endif
    }
    return (int)m_outTensorNumDetections->get_data<float>()[0];
}

int mbInterfaceATR::GetResultClasses(int i)
{
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
#ifdef NO_TRT
        LOG_F(ERROR, "Model type is TensorRT but it is not supported in this version!!!");
        return 0;
#else
        return m_trtAtrDetections[i].cls_id;
#endif
    }
    return (int)m_outTensorClasses->get_data<float>()[i];
}

float mbInterfaceATR::GetResultScores(int i)
{
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
#ifdef NO_TRT
        LOG_F(ERROR, "Model type is TensorRT but it is not supported in this version!!!");
        return 0;
#else
        return m_trtAtrDetections[i].score;
#endif
    }
    return m_outTensorScores->get_data<float>()[i];
}
std::vector<float> mbInterfaceATR::GetResultBoxes()
{
    if (m_modelType == e_ATR_MODEL_TYPE::ATR_MODEL_TYPE_TRT)
    {
#ifdef NO_TRT
        LOG_F(ERROR, "Model type is TensorRT but it is not supported in this version!!!");
        return 0;
#else
        return m_regressBBCoords;
#endif
    }
    return m_outTensorBB->get_data<float>();
}
