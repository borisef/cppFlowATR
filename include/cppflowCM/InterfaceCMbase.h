#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <cppflowATRInterface/Object_Detection_Types.h>

#include <string>

/**
 * @brief an abstract class for ColorModel specific implentation (tensor-rt engine / tensorflow) to derive from
 *
 */
class mbInterfaceCMbase
{
protected:
  int m_patchHeight = 128;
  int m_patchWidth = 128;
  int m_batchSize = 32;
  int m_numColors = 7;
  bool m_hardBatchSize = false;

public:
  float m_tileMargin = 0.2;

  //constructors/destructors
  mbInterfaceCMbase() : m_patchHeight(128), m_patchWidth(128), m_batchSize(32), m_numColors(7), m_hardBatchSize(false), m_tileMargin(0.2){};
  mbInterfaceCMbase(int h, int w, int nc, int bs, bool hbs) : m_patchHeight(h), m_patchWidth(w), m_batchSize(bs), m_numColors(nc), m_hardBatchSize(hbs), m_tileMargin(0.2){};
  virtual ~mbInterfaceCMbase(){};

  /**
   * @brief Load a new model, replacing the previous one (if any)
   *
   * @param modelPath path to the model file
   * @param ckptPath an optional path to a checpoint file to initialize weights from it
   * @param intensor an optional tensor name to use for network input, required if the input node is not provided by the model.
   * @param outtensor an optional tensor name to use for network output, required if the output node is not provided by the model
   *                  or if the model has more then one output node.
   * @return true on success
   * @return false on failure
   */
  virtual bool LoadNewModel(const char *modelPath, const char *ckptPath, const char *intensor, const char *outtensor) = 0;
  /**
   * @brief run color-detection-model on the given image
   *
   * @param img an RGB (8 bit/ch) image to feed the network (will be resized and normalized to net)
   * @return std::vector<float> per color probability (each index of the vector represent a color and the mapping is predefined)
   */
  virtual std::vector<float> RunRGBimage(cv::Mat img) = 0;
  /**
   * @brief  run color-detection-model on the given image
   *
   * @param ptr a path to a file
   * @return std::vector<float> per color probability (each index of the vector represent a color and the mapping is predefined)
   */
  virtual std::vector<float> RunRGBImgPath(const unsigned char *ptr) = 0;
  /**
   * @brief run color-detection-model on an patch of the given image
   *
   * @param img an RGB (8 bit/ch) image to crop a patch from it
   * @param bb pixel coordinates of a bounding-box in the image `img` to use for the color-detection-model
   * @return std::vector<float> per color probability (each index of the vector represent a color and the mapping is predefined)
   */
  virtual std::vector<float> RunImgBB(cv::Mat img, OD::OD_BoundingBox bb) = 0;
  /**
   * @brief color-detection-model on a set of patches in the given image
   *
   * @param img an RGB (8 bit/ch) image to crop a patch from it
   * @param co [inout] Object detection algorithm output - a list of object detection output objects that holds
   *                   the bounding-boxes coords on the given image to use for color-detection-model.
   *                   optionally, if `copyResults` is `true`, the result color for each detected object
   *                   is also stored on the objects of the given `od` list.
   * @param startInd the index on the `co` list for the first object to run color-detection-model on it.
   * @param stopInd the index on the `co` list for the first object to run color-detection-model on it.
   * @param copyResults if set to false, results are discarded. Otherwise, results are saved to the relevant
   *                    indices in the given `od` list.
   * @return true
   * @return false never //TODO: change return type to void or consider returning false on failure.
   */
  virtual bool RunImgWithCycleOutput(cv::Mat img, OD::OD_CycleOutput *co, int startInd, int stopInd, bool copyResults = true) = 0;

  /**
   * @brief run color-detection-model on an arbitrary input ignoring the results. Usd for warm-up.
   */
  virtual void IdleRun() = 0;

  /**
   * @brief convert color-model's id to object-detection's id
   *
   * @param cid color-model's id (network's returned value)
   * @return OD::e_OD_TargetColor object-detection generic id
   */
  OD::e_OD_TargetColor TargetColor(uint cid)
  {
    OD::e_OD_TargetColor ans = OD::e_OD_TargetColor::UNKNOWN_COLOR;
    // map color id to OD::e_OD_TargetColor
    static OD::e_OD_TargetColor cid_to_enum[] = {OD::e_OD_TargetColor::BLACK,   // 0
                                                 OD::e_OD_TargetColor::BLUE,    // 1
                                                 OD::e_OD_TargetColor::GRAY,    // 2
                                                 OD::e_OD_TargetColor::GREEN,   // 3
                                                 OD::e_OD_TargetColor::RED,     // 4
                                                 OD::e_OD_TargetColor::WHITE,   // 5
                                                 OD::e_OD_TargetColor::YELLOW}; // 6
    static int num_colors = sizeof(cid_to_enum) / sizeof(OD::e_OD_TargetColor); //num of colors in the array
    if (cid < num_colors)
    {
      ans = cid_to_enum[cid];
    }

#ifdef TEST_MODE
    // map color id to color-name
    static const char *cid_to_cname[] = {"black",   // 0
                                         "blue",    // 1
                                         "gray",    // 2
                                         "green",   // 3
                                         "red",     // 4
                                         "white",   // 5
                                         "yellow"}; // 6
    const char *color_name = "UNKNOWN_COLOR";
    if (ans != OD::e_OD_TargetColor::UNKNOWN_COLOR)
    {
      color_name = cid_to_cname[cid];
    }
    std::cout << "Color: " << color_name << std::endl;
#endif

    return ans;
  }
};
