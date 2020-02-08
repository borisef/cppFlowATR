#include <utils/odUtils.h>

using namespace cv;
using namespace std;


OD_CycleOutput* NewOD_CycleOutput(int maxNumOfObjects, int defaultImgID_output){
    
    OD_CycleOutput *co = new OD_CycleOutput(); // allocate empty cycle output buffer
    co->maxNumOfObjects = maxNumOfObjects;
    co->ImgID_output = defaultImgID_output;
    co->numOfObjects = 0;
    co->ObjectsArr = new OD_DetectionItem[co->maxNumOfObjects];
    return co;
}

void swap_OD_DetectionItem(OD_DetectionItem* xp, OD_DetectionItem * yp)  
{  
    OD_DetectionItem temp = *xp;  
    *xp = *yp;  
    *yp = temp;  
}  
void bubbleSort_OD_DetectionItem(OD_DetectionItem* arr, int n)  
{  
    int i, j;  
    for (i = 0; i < n-1; i++)      
      
    // Last i elements are already in place  
    for (j = 0; j < n-i-1; j++)  
        if (arr[j].tarScore < arr[j+1].tarScore)  
            swap_OD_DetectionItem(&arr[j], &arr[j+1]);  
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
        cout << "Color: white" << endl;
        return cv::Scalar(255,255,255);
        break;
    case OD::e_OD_TargetColor::BLACK:
        cout << "Color: black" << endl;
        return cv::Scalar(0,0,0);
        break;
    case OD::e_OD_TargetColor::GRAY:
        cout << "Color: gray" << endl;
        return cv::Scalar(100,100,100);
        break;
    case OD::e_OD_TargetColor::RED:
        cout << "Color: red" << endl;
        return cv::Scalar(255,0,0);
        break;
    case OD::e_OD_TargetColor::GREEN:
        cout << "Color: green" << endl;
        return cv::Scalar(0,255,0);
        break;
    case OD::e_OD_TargetColor::BLUE:
        cout << "Color: blue" << endl;
        return cv::Scalar(0,0,255);
        break;
    case OD::e_OD_TargetColor::YELLOW:
        cout << "Color: yellow" << endl;
        return cv::Scalar(255,255,0);
        break;
    }

    return cv::Scalar(0,255,255);

}