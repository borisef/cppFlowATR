///////////////////////////////////////////////////////////////////////////
// File Name:   Object_Detection_Types.h
//
// Description: Types Definitions for Object Detection algorithm
//
// Copyright (c) 2019(year of creation) Rafael Ltd. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef OBJECT_DETECTION_TYPES_H
#define OBJECT_DETECTION_TYPES_H


#pragma once
#pragma pack(1)

namespace OD
{

enum OD_ErrorCode
{
	OD_OK = 0,
	OD_FAILURE = 1,
	OD_INIT_FAIL = 2,
	OD_CANNOT_ALLOCATE_MEMORY = 3,
	OD_ILEGAL_INPUT = 4,
	OD_RESET_FAIL = 5,
	OD_WRONG_METRY_SIZE = 6
};

enum e_OD_ColorImageType
{
	COLOR = 0,
	BLACK_WHITE = 1,
	YUV422 = 2,
	RGB = 3,
	BGR = 4,
	YUV = 5,
	RGB_IMG_PATH = 6, 
	NV12 = 7
};

enum MB_MissionType //BE
{
	MATMON = 0,
	STATIC_CHASER = 1,
	DYNAMIC_CHASER = 2,
	ANALYZE_SAMPLE = 3

};


enum e_OD_TargetSubClass
{
	UNKNOWN_SUB_CLASS = 1,
	PRIVATE = 2,
	COMMERCIAL = 3,
	PICKUP = 4,
	TRUCK = 5,
	BUS = 6,
	VAN = 7,
	TRACKTOR = 8,
	OTHER_SUB_CLASS = 999
};


enum e_OD_TargetColor
{
	UNKNOWN_COLOR = 1,
	WHITE = 2,
	SILVER = 3,
	GRAY = 4,
	BLACK = 5,
	RED = 6,
	GREEN = 7,
	BLUE = 8,
	BROWN = 9,
	YELLOW = 10, //BE
	OTHER = 999
};

struct MB_Mission //BE
{
	MB_MissionType missionType;
	e_OD_TargetSubClass targetClas;
	e_OD_TargetColor targetColor;
};

struct OD_SupportData // extra data to support detection alg
{
	unsigned int imageHeight;
	unsigned int imageWidth;
	e_OD_ColorImageType colorType;
	float rangeInMeters;
	float cameraAngle;		//BE
	float cameraParams[10]; //BE
	float spare[3];
};

struct OD_InitParams
{
	const char *iniFilePath;   // path to ini file
	unsigned int numOfObjects; // max number of items to be returned
	OD_SupportData supportData;

	MB_Mission mbMission; //BE: task
};

struct OD_BoundingBox
{
	float x1; // top left point
	float x2; // top right point
	float y1; //bottom left point
	float y2; //bottom right point
};

enum e_OD_TargetClass
{
	UNKNOWN_CLASS = 1,
	VEHICLE = 2,
	PERSON = 3,
	OTHER_CLASS = 999
};

// enum e_OD_TargetSubClass_MB //BE
// {
// 	PERSON			= 1,
// 	CAR				= 5,
// 	BICYCLE			= 10,
// 	MOTORCYCLE		= 11,
// 	BUS				= 12,
// 	TRUCK			= 13,
// 	VAN				= 14,
// 	JEEP 			= 15,
// 	PICKUP_OPEN 	= 16,
// 	PICKUP_CLOSED 	= 20,
// 	FORKLIFT 		= 17,
// 	TRACKTOR 		= 18,
// 	STATION 		= 19,
// 	OTHER			= 999
// };

struct OD_DetectionItem
{
	OD_BoundingBox tarBoundingBox;
	e_OD_TargetClass tarClass;
	e_OD_TargetSubClass tarSubClass;
	e_OD_TargetColor tarColor;
	float tarScore;
	float tarColorScore;  //BE
	float occlusionScore; //BE
						  //OD_DetectionItem();
						  //void copyData(OD_DetectionItem tocopy);
};

struct OD_CycleInput
{
	unsigned int ImgID_input;
	const unsigned char *ptr; // pointer to picture buffer
							  //OD_CycleInput(int ii):ImgID_input(ii){ptr=nullptr;}
							  //OD_CycleInput(int ii, const unsigned char *pp ):ImgID_input(ii){ptr=pp;}
};

struct OD_CycleOutput
{
	unsigned int ImgID_output;
	unsigned int numOfObjects;
	unsigned int maxNumOfObjects;
	OD_DetectionItem *ObjectsArr;

	//constructors
	// OD_CycleOutput(OD_CycleOutput & tocopy);
	// OD_CycleOutput(int maxTargets);
	// OD_CycleOutput();
};
} //namespace OD
#pragma pack()
#endif // OBJECT_DETECTION_TYPES_H