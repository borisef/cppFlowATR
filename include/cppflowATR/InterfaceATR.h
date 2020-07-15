#pragma once

#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <cppflow/Tensor.h>
#include <cppflow/Model.h>

#ifndef NO_TRT
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <utils/trt_buffers.h>
#include <utils/trt_loguru_wrapper.h>
#endif


enum e_ATR_MODEL_TYPE
{
    ATR_MODEL_TYPE_UNKNOWN = 0,
    ATR_MODEL_TYPE_TF,
    ATR_MODEL_TYPE_TRT
};

class mbInterfaceATR
{
  protected:
    Model* m_model;
    Tensor* m_outTensorNumDetections;
    Tensor* m_outTensorScores;
    Tensor* m_outTensorBB;
    Tensor* m_outTensorClasses;
    Tensor* m_inpName;
    e_ATR_MODEL_TYPE m_modelType;

#ifndef NO_TRT
    //TensorRT related class members:
    struct trtAtrDetection
    {
      float score;
      int cls_id;
      float bb_coords[4]; //x1,y1,x2,y2

      trtAtrDetection(const float _bb_coords[4], const int _cls_id, const float _score)
      {
        score = _score;
        cls_id = _cls_id;
        //memcpy()
        std::copy_n(_bb_coords, 4, bb_coords);
      }

      // sort by score
      bool operator<(const trtAtrDetection& other) const
      {
        if (score != other.score)
          return score < other.score;
        // prevent different objects of being cosidered the same if we are using std::set (or similar containers)
        if (cls_id != other.cls_id)
          return cls_id < other.cls_id;
        return ( bb_coords[0] < other.bb_coords[0] ) || ( bb_coords[1] < other.bb_coords[1] ) ||
               ( bb_coords[2] < other.bb_coords[2] ) || ( bb_coords[3] < other.bb_coords[3] );
      }
      // for descending order
      bool operator>(const trtAtrDetection& other) const
      {
        return other.operator<(*this);
      }
    }; //struct trtAtrDetection

    struct inputDims
    {
      int m_numChannels;
      int m_height;
      int m_width;
    } m_inputDims;
    TRTLoguruWrapper m_logger;
    nvinfer1::ICudaEngine *m_engine;
    nvinfer1::IExecutionContext *m_context;
    cudaStream_t m_stream;
    cudaEvent_t m_start;
    cudaEvent_t m_end;
    BufferManager *m_bufferManager;
    int m_DLACore;
    int m_batchSizeTRT;
    bool m_isInitialized;
    std::map<int, std::string> m_inTensors;
    std::map<int, std::string> m_outTensors;
    std::vector<float> m_regressBBCoords;
    //std::vector<float> m_perClassScores;
    std::vector<trtAtrDetection> m_trtAtrDetections;
    int m_numClasses; // this includes the background class which have no BB
    int m_numProposals;
#endif

  public:
    cv::Mat m_keepImg;
    cv::Mat GetKeepImg(){return m_keepImg;}

    //constructors
    mbInterfaceATR();
    ~mbInterfaceATR();
    void InitTRTStuff();
    void DestroyTRTStuff();
    bool LoadNewModel(const char* modelPath);
    bool LoadNewTRTModel(const char* modelPath);
    bool imageToTRTInputBuffer(const std::vector<uint8_t> &img_data);
    bool doTRTInference();
    bool getTRTOutput();
    int RunRGBimage(cv::Mat img);
    int RunRGBVector(const unsigned char *ptr, int height, int width, float resize_factor = 1.0f);
    int RunRGBVector(std::vector<uint8_t > &img_data, int height, int width, float resize_factor = 1.0f);
    int RunRGBImgPath(const unsigned char *ptr, float resize_factor = 1.0f);
    int RunRawImage(const unsigned char *ptr, int height, int width);
    int RunRawImageFast(const unsigned char *ptr, int height, int width, int colorType, float resize_factor = 1.0f);
    int GetResultNumDetections();
    int GetResultClasses(int i);
    float GetResultScores(int i);
    std::vector<float>  GetResultBoxes(); // array of: r,c,r,c,r,c,r,c,... (BB1.y1, BB1.x1, BB1.y2, BB1.x2, BB2.y1, BB2.x1, BB2.y2, BB2.x2, ...)
};
