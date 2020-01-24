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
