#include <utils/odUtils.h>

using namespace cv;
using namespace std;



	// ATR_PERSON		= 1,
	// ATR_CAR				= 5,
	// ATR_BICYCLE			= 10,
	// ATR_MOTORCYCLE		= 11,
	// ATR_BUS				= 12,
	// ATR_TRUCK			= 13,
	// ATR_VAN				= 14,
	// ATR_JEEP 			= 15,
	// ATR_PICKUP_OPEN 	= 16,
	// ATR_PICKUP_CLOSED 	= 20,
	// ATR_FORKLIFT 		= 17,
	// ATR_TRACKTOR 		= 18,
	// ATR_STATION 		= 19,
	// ATR_OTHER			= 999


// enum e_OD_TargetClass
// {
// 	UNKNOWN_CLASS = 1,
// 	VEHICLE = 2,
// 	PERSON = 3,
// 	OTHER_CLASS = 999
// };

// enum e_OD_TargetSubClass
// {
// 	UNKNOWN_SUB_CLASS = 1,//used for human
// 	PRIVATE = 2,
// 	COMMERCIAL = 3,
// 	PICKUP = 4,
// 	TRUCK = 5,
// 	BUS = 6,
// 	VAN = 7,
// 	TRACKTOR = 8,
// 	OTHER_SUB_CLASS = 999 // used for any 
// };


std::map<ATR_TargetSubClass_MB, e_OD_TargetSubClass> mapOfATR2SubClass = {
    {ATR_TargetSubClass_MB::ATR_PERSON,OD::e_OD_TargetSubClass::UNKNOWN_SUB_CLASS},
     {ATR_TargetSubClass_MB::ATR_BICYCLE,OD::e_OD_TargetSubClass::UNKNOWN_SUB_CLASS},
      {ATR_TargetSubClass_MB::ATR_MOTORCYCLE,OD::e_OD_TargetSubClass::UNKNOWN_SUB_CLASS},
     {ATR_TargetSubClass_MB::ATR_CAR,OD::e_OD_TargetSubClass::PRIVATE},
     {ATR_TargetSubClass_MB::ATR_BUS,OD::e_OD_TargetSubClass::BUS},
     {ATR_TargetSubClass_MB::ATR_TRUCK,OD::e_OD_TargetSubClass::TRUCK},
     {ATR_TargetSubClass_MB::ATR_VAN,OD::e_OD_TargetSubClass::VAN},
     {ATR_TargetSubClass_MB::ATR_JEEP,OD::e_OD_TargetSubClass::COMMERCIAL},
     {ATR_TargetSubClass_MB::ATR_PICKUP,OD::e_OD_TargetSubClass::PICKUP},
     {ATR_TargetSubClass_MB::ATR_PICKUP_OPEN,OD::e_OD_TargetSubClass::PICKUP},
     {ATR_TargetSubClass_MB::ATR_PICKUP_CLOSED,OD::e_OD_TargetSubClass::COMMERCIAL},
     {ATR_TargetSubClass_MB::ATR_FORKLIFT,OD::e_OD_TargetSubClass::TRACKTOR},//?
     {ATR_TargetSubClass_MB::ATR_TRACKTOR,OD::e_OD_TargetSubClass::TRACKTOR},
     {ATR_TargetSubClass_MB::ATR_STATION,OD::e_OD_TargetSubClass::COMMERCIAL},
     {ATR_TargetSubClass_MB::ATR_OTHER,OD::e_OD_TargetSubClass::OTHER_SUB_CLASS}
     };

std::map<ATR_TargetSubClass_MB, e_OD_TargetClass> mapOfATR2Class = {
    {ATR_TargetSubClass_MB::ATR_PERSON,OD::e_OD_TargetClass::PERSON},
     {ATR_TargetSubClass_MB::ATR_BICYCLE,OD::e_OD_TargetClass::OTHER_CLASS},
     {ATR_TargetSubClass_MB::ATR_MOTORCYCLE,OD::e_OD_TargetClass::OTHER_CLASS},
     {ATR_TargetSubClass_MB::ATR_CAR,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_BUS,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_TRUCK,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_VAN,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_JEEP,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_PICKUP,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_PICKUP_OPEN,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_PICKUP_CLOSED,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_FORKLIFT,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_TRACKTOR,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_STATION,OD::e_OD_TargetClass::VEHICLE},
     {ATR_TargetSubClass_MB::ATR_OTHER,OD::e_OD_TargetClass::OTHER_CLASS}
     };


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

std::map<e_OD_TargetClass, char *> mapOfClasses = {
    {UNKNOWN_CLASS, "UNKNOWN_CLASS"},
    {VEHICLE, "VEHICLE"},
    {PERSON, "PERSON"},
    {OTHER_CLASS, "OTHER_CLASS"}
};

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

std::string GetColorString(OD::e_OD_TargetColor color_id)
{
    switch (color_id)
    {
    case OD::e_OD_TargetColor::WHITE:
        return(std::string("white"));
        break;
    case OD::e_OD_TargetColor::BLACK:
        return(std::string("black"));
        break;
    case OD::e_OD_TargetColor::GRAY:
        return(std::string("gray"));
        break;
    case OD::e_OD_TargetColor::RED:
        return(std::string("red"));
        break;
    case OD::e_OD_TargetColor::GREEN:
        return(std::string("green"));
        break;
    case OD::e_OD_TargetColor::BLUE:
        return(std::string("blue"));
        break;
    case OD::e_OD_TargetColor::YELLOW:
        return(std::string("yellow"));
        break;
    }

    return(std::string("weird"));

}

std::string GetStringInitParams(OD::OD_InitParams ip)
{
    std::string mystr = "OD_InitParams: ";
    mystr.append(ip.iniFilePath).append("\n");
    mystr.append("ip.mbMission.missionType = ").append(mapOfmissions[ip.mbMission.missionType]).append("\n");
    mystr.append("ip.supportData.colorType = ").append(std::to_string(ip.supportData.colorType)).append("\n");
    mystr.append("ip.supportData.imageHeight = ").append(std::to_string(ip.supportData.imageHeight)).append("\n");
    mystr.append("ip.supportData.imageWidth = ").append(std::to_string(ip.supportData.imageWidth)).append("\n");
    mystr.append("ip.mbMission.targetSubClass = ").append(std::to_string(ip.mbMission.targetSubClass)).append("\n");

    return mystr;
}

std::string BB2LogString(OD_BoundingBox bb)
{
    std::string mystr = "(";
    mystr.append(std::to_string(int(bb.x1))).append(",");
    mystr.append(std::to_string(int(bb.x2))).append(",");
    mystr.append(std::to_string(int(bb.y1))).append(",");
    mystr.append(std::to_string(int(bb.y2))).append(")");
    return mystr;
}

std::string DetectionItem2LogString(OD_DetectionItem di)
{
    std::string mystr = "";
    mystr.append(GetFromMapOfClasses(di.tarClass)).append("/");
    mystr.append(GetFromMapOfSubClasses(di.tarSubClass)); 
    mystr.append("(").append(std::to_string(di.tarClass)).append("/");
    mystr.append(std::to_string(di.tarSubClass)).append("),");
    mystr.append("bb:").append(BB2LogString(di.tarBoundingBox)).append(",");
  //  mystr.append(mapOfClasses[di.tarClass]).append(",");
    mystr.append("sc:").append(std::to_string(int(100*di.tarScore))).append(",");
   // mystr.append(mapOfcolors[di.tarColor]).append(",");
    mystr.append(GetColorString(di.tarColor)).append("(");
    mystr.append(std::to_string(di.tarColor)).append("),");
    mystr.append("col_sc:").append(std::to_string(int(100*(di.tarColorScore)))).append("");
    mystr.append("\n");
    return mystr;
}
std::string CycleOutput2LogString(OD_CycleOutput* co)
{
    std::string mystr = "OD_CycleOutput:\n ";
    mystr.append("numOfObjects=").append(std::to_string(co->numOfObjects)).append("\n");
    for (size_t i = 0; i < co->numOfObjects; i++)
    {
        mystr.append("OD_DetectionItem:").append(std::to_string(i)).append("\n"); 
        mystr.append(DetectionItem2LogString(co->ObjectsArr[i]));
    }
    
    //mystr.append("TODO=").append("TODO").append("\n");

    return mystr;

}

std::string GetFromMapOfClasses(e_OD_TargetClass cl)
{

    char* rrr = mapOfClasses[cl];
    if(rrr == 0x0)
        return ("UNKNOW_CLASS");    
    return (std::string(rrr));

}
std::string GetFromMapOfSubClasses(e_OD_TargetSubClass scl)
{

    char* rrr = mapOfSubclasses[scl];
    if(rrr == 0x0)
        return ("UNKNOW_SUBCLASS");
    return (std::string(rrr));

}

void MapATR_Classes(ATR_TargetSubClass_MB inClass, OD::e_OD_TargetClass& outClass, OD::e_OD_TargetSubClass& outSubClass)
{
    outClass = mapOfATR2Class[inClass];
    outSubClass = mapOfATR2SubClass[inClass];

    if(outClass==0)
        outClass = OD::e_OD_TargetClass::UNKNOWN_CLASS;

    if(outSubClass==0)
        outSubClass = OD::e_OD_TargetSubClass::UNKNOWN_SUB_CLASS;
    
};

int SqueezeCycleOutputInplace(OD_CycleOutput* co)
{
    int N = co->numOfObjects;
    float eps = 0.001;
    int move2=0;
     for (size_t i = 0; i < N; i++)
     {
        if(co->ObjectsArr[i].tarScore > eps)
        {
            if(move2<i)
            {
                //move i into move2
                co->ObjectsArr[move2].tarScore=co->ObjectsArr[i].tarScore;
                co->ObjectsArr[move2].tarColorScore=co->ObjectsArr[i].tarColorScore;
                co->ObjectsArr[move2].occlusionScore=co->ObjectsArr[i].occlusionScore;
                co->ObjectsArr[move2].tarClass=co->ObjectsArr[i].tarClass;
                co->ObjectsArr[move2].tarBoundingBox=co->ObjectsArr[i].tarBoundingBox;
                co->ObjectsArr[move2].tarColor=co->ObjectsArr[i].tarColor;
                co->ObjectsArr[move2].tarSubClass=co->ObjectsArr[i].tarSubClass;
            }
            move2=move2+1;
           
        }
        else
        {
            co->numOfObjects= co->numOfObjects-1;
        }
        
     }
    
    return(co->numOfObjects);
}

int FilterCycleOutputByClassNoSqueeze(OD_CycleOutput* co, e_OD_TargetClass class2remove)
{
    int N = co->numOfObjects;
     for (size_t i = 0; i < N; i++)
     {
         if(co->ObjectsArr[i].tarClass==class2remove)
            co->ObjectsArr[i].tarScore = 0.0f;
     }
}

int FilterCycleOutputBySubClassNoSqueeze(OD_CycleOutput* co, e_OD_TargetSubClass subclass2remove)
{
    int N = co->numOfObjects;
     for (size_t i = 1; i < N; i++)
     {
         if(co->ObjectsArr[i].tarSubClass==subclass2remove)
            co->ObjectsArr[i].tarScore = 0.0f;
     }
}