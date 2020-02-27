#include <utils/odUtils.h>

using namespace cv;
using namespace std;

std::map<OD::e_OD_TargetColor, char *> mapOfcolors = {
    {OD::e_OD_TargetColor::BLACK, "black"},
    {OD::e_OD_TargetColor::BLUE, "blue"},
    {OD::e_OD_TargetColor::YELLOW, "yellow"},
    {OD::e_OD_TargetColor::GRAY, "gray"},
    {OD::e_OD_TargetColor::GREEN, "green"},
    {OD::e_OD_TargetColor::RED, "red"},
    {OD::e_OD_TargetColor::WHITE, "white"},
    {OD::e_OD_TargetColor::SILVER, "silver"},
    {OD::e_OD_TargetColor::BROWN, "brown"}};

std::map<MB_MissionType, char *> mapOfmissions = {
    {MB_MissionType::ANALYZE_SAMPLE, "ANALYZE_SAMPLE"},
    {MB_MissionType::MATMON, "MATMON"},
    {MB_MissionType::STATIC_CHASER, "STATIC_CHASER"},
    {MB_MissionType::DYNAMIC_CHASER, "DYNAMIC_CHASER"}};

std::map<e_OD_ColorImageType, char *> mapOfImgTypes = {
    {COLOR, "COLOR"},
    {BLACK_WHITE, "BLACK_WHITE"},
    {YUV422, "YUV422"},
    {RGB, "RGB"},
    {BGR, "BGR"},
    {YUV, "YUV"},
    {RGB_IMG_PATH, "RGB_IMG_PATH"},
    {NV12, "NV12"}};

std::map<e_OD_TargetSubClass, char *> mapOfSubclasses = {
    {UNKNOWN_SUB_CLASS, "UNKNOWN_SUB_CLASS"},
    {PRIVATE, "PRIVATE"},
    {COMMERCIAL, "COMMERCIAL"},
    {PICKUP, "PICKUP"},
    {TRUCK, "TRUCK"},
    {BUS, "BUS"},
    {VAN, "VAN"},
    {TRACKTOR, "TRACKTOR"},
    {OTHER_SUB_CLASS, "OTHER_SUB_CLASS"}};

OD_CycleOutput *NewOD_CycleOutput(int maxNumOfObjects, int defaultImgID_output)
{

    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    co->maxNumOfObjects = maxNumOfObjects;
    co->ImgID_output = defaultImgID_output;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];
    return co;
}

void swap_OD_DetectionItem(OD_DetectionItem *xp, OD_DetectionItem *yp)
{
    OD_DetectionItem temp = *xp;
    *xp = *yp;
    *yp = temp;
}
void bubbleSort_OD_DetectionItem(OD_DetectionItem *arr, int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)

        // Last i elements are already in place
        for (j = 0; j < n - i - 1; j++)
            if (arr[j].tarScore < arr[j + 1].tarScore)
                swap_OD_DetectionItem(&arr[j], &arr[j + 1]);
}

void PrintColor(int color_id)
{
    switch (color_id)
    {
    case 0:
        cout << "Color: white" << endl;
        break;
    case 1:
        cout << "Color: black" << endl;
        break;
    case 2:
        cout << "Color: gray" << endl;
        break;
    case 3:
        cout << "Color: red" << endl;
        break;
    case 4:
        cout << "Color: green" << endl;
        break;
    case 5:
        cout << "Color: blue" << endl;
        break;
    case 6:
        cout << "Color: yellow" << endl;
        break;
    }
}

cv::Scalar GetColor2Draw(OD::e_OD_TargetColor color_id)
{
    switch (color_id)
    {
    case OD::e_OD_TargetColor::WHITE:
#ifdef TEST_MODE
        cout << "Color: white" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(255, 255, 255);
        break;
    case OD::e_OD_TargetColor::BLACK:
#ifdef TEST_MODE

        cout << "Color: black" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(0, 0, 0);
        break;
    case OD::e_OD_TargetColor::GRAY:
#ifdef TEST_MODE

        cout << "Color: gray" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(100, 100, 100);
        break;
    case OD::e_OD_TargetColor::RED:
#ifdef TEST_MODE

        cout << "Color: red" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(255, 0, 0);
        break;
    case OD::e_OD_TargetColor::GREEN:
#ifdef TEST_MODE

        cout << "Color: green" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(0, 255, 0);
        break;
    case OD::e_OD_TargetColor::BLUE:
#ifdef TEST_MODE

        cout << "Color: blue" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(0, 0, 255);
        break;
    case OD::e_OD_TargetColor::YELLOW:
#ifdef TEST_MODE
        cout << "Color: yellow" << endl;
#endif //#ifdef TEST_MODE

        return cv::Scalar(255, 255, 0);
        break;
    }

    return cv::Scalar(0, 255, 255);
}

std::string GetStringInitParams(OD::OD_InitParams ip)
{
    std::string mystr = "OD_InitParams: ";
    mystr.append(ip.iniFilePath).append("\n");
    mystr.append("ip.mbMission.missionType = ").append(mapOfmissions[ip.mbMission.missionType]).append("\n");
    mystr.append("ip.supportData.colorType = ").append(std::to_string(ip.supportData.colorType)).append("\n");
    mystr.append("ip.supportData.imageHeight = ").append(std::to_string(ip.supportData.imageHeight)).append("\n");
    mystr.append("ip.supportData.imageWidth = ").append(std::to_string(ip.supportData.imageWidth)).append("\n");
    mystr.append("ip.mbMission.targetClas = ").append(std::to_string(ip.mbMission.targetClas)).append("\n");

    return mystr;
}