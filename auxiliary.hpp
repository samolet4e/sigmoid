#ifndef AUXILIARY
#define AUXILIARY

#include <opencv2/opencv.hpp>

typedef double Real;

#define PI (4. * atan(1.))

typedef struct _fixation {
	Real timeStamp, duration;
//	Real x[2]; // relative coords
	cv::Point2d X;
} fixation;

typedef struct _gaze {
    Real timeStamp;
    cv::Point2d X;
} gaze;

typedef struct _saccade {
    fixation fPoint;
    std::vector <gaze> gPoint;
} saccade;

typedef struct _datum {
	cv::Point2d X;
} datum;

int readFixations(std::vector <fixation> &fPoint);
int readGazes(std::vector <gaze> &gPoint);
int doDeriv(cv::Mat& dfdl, Real x, cv::Mat& lambda);
Real funcToFit(Real x, cv::Mat& lambda);
Real funcPrimeToFit(Real x, cv::Mat& lambda);
int makeFit(int dataSize, std::vector <datum>& data, cv::Mat& lambda);
Real rsd(int dataSize, std::vector <datum>& data, cv::Mat& lambda);
int loadData(std::vector <saccade>& sLine, std::vector <datum>& data);

#endif
