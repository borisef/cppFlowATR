#include <cppflowATRInterface/Object_Detection_Types.h>

void OD_DetectionItem::copyData(OD_DetectionItem tocopy)
{
    tarBoundingBox = tocopy.tarBoundingBox;
	 tarClass = tocopy.tarClass;
	 tarSubClass = tocopy.tarSubClass;
	 tarColor = tocopy.tarColor;
	 tarScore = tocopy.tarScore;
	 tarColorScore = tocopy.tarColorScore;  //BE
	 occlusionScore = tocopy.occlusionScore; //BE
}


OD_CycleOutput::OD_CycleOutput(OD_CycleOutput & tocopy)
{
    ImgID_output = tocopy.ImgID_output; 
    numOfObjects = tocopy.numOfObjects;
    maxNumOfObjects = tocopy.maxNumOfObjects;
    ObjectsArr = new OD_DetectionItem[maxNumOfObjects]; //TODO deep copy
    for (int i=0;i<maxNumOfObjects;i++)
    {
        ObjectsArr[i].copyData(tocopy.ObjectsArr[i]);
    }
}

OD_CycleOutput::OD_CycleOutput (int maxTargets = 100)
{
    ImgID_output = -1;
	numOfObjects = 0;
    maxNumOfObjects = maxTargets;
	ObjectsArr = new OD_DetectionItem[maxTargets];
}

OD_DetectionItem::OD_DetectionItem()
{
	//tarBoundingBox;
	tarClass = e_OD_TargetClass::UNKNOWN_CLASS;
	tarSubClass = e_OD_TargetSubClass::UNKNOWN_SUB_CLASS;
	tarColor = e_OD_TargetColor::UNKNOWN_COLOR;
	tarScore = 0;
	tarColorScore = 0 ; 
	occlusionScore = 0; 
};
