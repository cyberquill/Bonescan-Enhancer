#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <omp.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat scan,pScan;
    int i,j,val;
    string windowName;
    cout<<"\n\n";
    cout<<"\n\t=================================================================================\n";
    cout<<"\t     TIME    |   STEP   |\t\t\tDESCRIPTION";
    cout<<"\n\t=================================================================================\n";

    //=================================================================================

    auto gStart = chrono::steady_clock::now();
    scan = imread("Step1 - original scan.png",IMREAD_GRAYSCALE);
    copyMakeBorder(scan,pScan,1,1,1,1,BORDER_CONSTANT,Scalar(0));
    auto stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-1  |  "<<"Read the input image.\n";

    //=================================================================================

    Mat sharpenedScan(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    Mat sobelOfScan(scan.rows,scan.cols,CV_8UC1,Scalar(0)),pSobelOfScan;
    #pragma omp parallel sections private(i,j,val,stepEnd)
    {
        #pragma omp section
        {
            for(i=1;i<pScan.rows-1;i++)
                for(j=1;j<pScan.cols-1;j++)
                {
                    val=(pScan.at<uchar>(i,j)*9) - (pScan.at<uchar>(i-1,j-1)+pScan.at<uchar>(i-1,j)+pScan.at<uchar>(i-1,j+1)+pScan.at<uchar>(i,j-1)+pScan.at<uchar>(i,j+1)+pScan.at<uchar>(i+1,j-1)+pScan.at<uchar>(i+1,j)+pScan.at<uchar>(i+1,j+1));
                    if(val<0)   val=0;
                    if(val>255) val=255;
                    sharpenedScan.at<uchar>(i-1,j-1) = val;
                }
            imwrite("Step2 - Sharpening by Laplacian.png",sharpenedScan);
            stepEnd = chrono::steady_clock::now();
            cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
            cout<<"  |  Step-2  |  "<<"Obtain Sharpened Version of Image, using Laplacian.\n";
        }
        #pragma omp section
        {
            int Gx, Gy;
            for(i=1;i<pScan.rows-1;i++)
                for(j=1;j<pScan.cols-1;j++)
                {
                    Gx=(pScan.at<uchar>(i-1,j+1)+2*pScan.at<uchar>(i,j+1)+pScan.at<uchar>(i+1,j+1))-(pScan.at<uchar>(i-1,j-1)+2*pScan.at<uchar>(i,j-1)+pScan.at<uchar>(i+1,j-1));
                    Gy=(pScan.at<uchar>(i+1,j-1)+2*pScan.at<uchar>(i+1,j)+pScan.at<uchar>(i+1,j+1))-(pScan.at<uchar>(i-1,j-1)+2*pScan.at<uchar>(i-1,j)+pScan.at<uchar>(i-1,j+1));
                    val=abs(Gx)+abs(Gy);
                    if(val<0)   val=0;
                    if(val>255) val=255;
                    sobelOfScan.at<uchar>(i-1,j-1)=val;
                }
            imwrite("Step3 - Sobel of Scan.png",sobelOfScan);
            copyMakeBorder(sobelOfScan,pSobelOfScan,2,2,2,2,BORDER_CONSTANT,Scalar(0));
            stepEnd = chrono::steady_clock::now();
            cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
            cout<<"  |  Step-3  |  "<<"Sobel operator\'s result on the input image.\n";
        }
    }
    /*
    Mat sharpenedScan(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    //#pragma omp parallel for private(j,val) collapse(2)
    for(i=1;i<pScan.rows-1;i++)
        for(j=1;j<pScan.cols-1;j++)
        {
            val=(pScan.at<uchar>(i,j)*9) - (pScan.at<uchar>(i-1,j-1)+pScan.at<uchar>(i-1,j)+pScan.at<uchar>(i-1,j+1)+pScan.at<uchar>(i,j-1)+pScan.at<uchar>(i,j+1)+pScan.at<uchar>(i+1,j-1)+pScan.at<uchar>(i+1,j)+pScan.at<uchar>(i+1,j+1));
            if(val<0)   val=0;
            if(val>255) val=255;
            sharpenedScan.at<uchar>(i-1,j-1) = val;
        }
    imwrite("Step2 - Sharpening by Laplacian.png",sharpenedScan);
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-2  |  "<<"Obtain Sharpened Version of Image, using Laplacian.\n";

    //=================================================================================

    Mat sobelOfScan(scan.rows,scan.cols,CV_8UC1,Scalar(0)),pSobelOfScan;
    int Gx, Gy;
    //#pragma omp parallel for private(j,val,Gx,Gy) collapse(2)
    for(i=1;i<pScan.rows-1;i++)
        for(j=1;j<pScan.cols-1;j++)
        {
            Gx=(pScan.at<uchar>(i-1,j+1)+2*pScan.at<uchar>(i,j+1)+pScan.at<uchar>(i+1,j+1))-(pScan.at<uchar>(i-1,j-1)+2*pScan.at<uchar>(i,j-1)+pScan.at<uchar>(i+1,j-1));
            Gy=(pScan.at<uchar>(i+1,j-1)+2*pScan.at<uchar>(i+1,j)+pScan.at<uchar>(i+1,j+1))-(pScan.at<uchar>(i-1,j-1)+2*pScan.at<uchar>(i-1,j)+pScan.at<uchar>(i-1,j+1));
            val=abs(Gx)+abs(Gy);
            if(val<0)   val=0;
            if(val>255) val=255;
            sobelOfScan.at<uchar>(i-1,j-1)=val;
        }
    imwrite("Step3 - Sobel of Scan.png",sobelOfScan);
    copyMakeBorder(sobelOfScan,pSobelOfScan,2,2,2,2,BORDER_CONSTANT,Scalar(0));
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-3  |  "<<"Sobel operator\'s result on the input image.\n";
    */
    //=================================================================================

    Mat smoothSobel(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    #pragma omp parallel for private(j,val) collapse(2)
    for(i=2;i<pSobelOfScan.rows-2;i++)
        for(j=2;j<pSobelOfScan.cols-2;j++)
        {
            val=0;
            for(int k=-2;k<=2;k++)
                for(int l=-2;l<=2;l++)
                    val+=pSobelOfScan.at<uchar>(i+k,j+l);
            val/=25;
            if(val<0)   val=0;
            if(val>255) val=255;
            smoothSobel.at<uchar>(i-2,j-2)=val;
        }
    imwrite("Step4 - Smoothed Sobel of Scan.png",smoothSobel);
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-4  |  "<<"Smoothed Sobel operator\'s result on the input image.\n";

    //=================================================================================

    Mat scanMask(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    #pragma omp parallel for private(j,val) collapse(2)
    for(i=0;i<scan.rows;i++)
        for(j=0;j<scan.cols;j++)
        {
            val=sharpenedScan.at<uchar>(i,j)*smoothSobel.at<uchar>(i,j);
            val=val/150;
            if(val<0)   val=0;
            if(val>255) val=255;
            scanMask.at<uchar>(i,j)=val;
        }
    imwrite("Step5 - Scan Mask.png",scanMask);
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-5  |  "<<"Obtained Final Image Mask.\n";

    //=================================================================================

    Mat enhancedScan(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    #pragma omp parallel for private(j,val) collapse(2)
    for(i=0;i<scan.rows;i++)
        for(j=0;j<scan.cols;j++)
        {
            val=scan.at<uchar>(i,j)+scanMask.at<uchar>(i,j);
            if(val<0)   val=0;
            if(val>255) val=255;
            enhancedScan.at<uchar>(i,j)=val;
        }
    imwrite("Step6 - Enhanced Scan.png",enhancedScan);
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-6  |  "<<"Rendering the Enhanced Scan.\n";

    //=================================================================================

    Mat powerTransform(scan.rows,scan.cols,CV_8UC1,Scalar(0));
    #pragma omp parallel for private(j,val) collapse(2)
    for(i=0;i<scan.rows;i++)
        for(j=0;j<scan.cols;j++)
        {
            val=15*pow(enhancedScan.at<uchar>(i,j),0.5);
            if(val<0)   val=0;
            if(val>255) val=255;
            powerTransform.at<uchar>(i,j)=val;
        }
    imwrite("Step7 - Power Law Transform Result.png",powerTransform);
    stepEnd = chrono::steady_clock::now();
    cout<<"\t"<<setfill(' ')<<setw(11)<<chrono::duration_cast<chrono::nanoseconds>(stepEnd-gStart).count();
    cout<<"  |  Step-7  |  "<<"Performing the power-law transformation.";

    //=================================================================================

    cout<<"\n\t=================================================================================\n";
    auto gEnd = chrono::steady_clock::now();
    cout<<"\t\t   PARALLEL EXECUTION TIME:\t"<<chrono::duration_cast<chrono::nanoseconds>(gEnd-gStart).count()<<"  nano-seconds";
    cout<<"\n\t=================================================================================\n";
    cout<<"\n\n";
    return 0;
}
