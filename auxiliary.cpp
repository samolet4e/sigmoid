#include "auxiliary.hpp"
#include <fstream>
#define SAC 70

int readFixations(std::vector <fixation> &fPoint) {

    Real startTime = 0.;

	std::ifstream myFile("fixations.txt");
	std::string line;

	bool firstLine = true;
	while (getline(myFile, line)) {
		std::stringstream ss(line);
		Real t, dt; // timeStamp, duration
		Real x[2]; // relative coordinates
		ss >> t >> dt >> x[0] >> x[1];
		if (firstLine == true) { startTime = t; firstLine = false; }
		t -= startTime;
		fixation blobT = (fixation){ .timeStamp = t, .duration = dt / 1000.f, .X = {x[0], 1. - x[1]} }; // Initialize the structure by compound literal (fixation). It is much safer, yet it might be omitted.
		fPoint.push_back(blobT);
	}//while

	myFile.close();

	return 0;
}//readFixations

int readGazes(std::vector <gaze> &gPoint) {

    Real startTime = 0.;

	std::ifstream myFile("gaze.txt");
	std::string line;

	bool firstLine = true;
    Real t, t_1 = 0.; // timeStamp
	while (getline(myFile, line)) {
		std::stringstream ss(line);
		Real x[2]; // relative coordinates
		ss >> t >> x[0] >> x[1];
		if (firstLine == true) { startTime = t; firstLine = false; }
		t -= startTime;
		if (t != t_1) {
            gaze blobT = (gaze){ .timeStamp = t, .X = {x[0], 1. - x[1]} };
            gPoint.push_back(blobT);
		}//if
		// Since there are numerous redundancies, make sure no identical gaze point passes through!
		t_1 = t;
	}//while

	myFile.close();

    return 0;
}//readGazes

int doDeriv(cv::Mat& dfdl, Real x, cv::Mat& lambda) {
Real A = lambda.at<Real>(0), a = lambda.at<Real>(1), b = lambda.at<Real>(2);

    dfdl.at<Real>(0) = 1. / (1. + exp((a - x) / b));
    dfdl.at<Real>(1) = -1. * A / b * exp((a - x) / b) / pow(1. + exp((a - x) / b), 2.);
    dfdl.at<Real>(2) = A / (b * b) * (a - x) * exp((a - x) / b) / pow(1. + exp((a - x) / b), 2.);

    return 0;
}//doDeriv

Real funcToFit(Real x, cv::Mat& lambda) {
Real A = lambda.at<Real>(0), a = lambda.at<Real>(1), b = lambda.at<Real>(2);

    return A / (1. + exp((a - x) / b));

}//funcToFit

Real funcPrimeToFit(Real x, cv::Mat& lambda) {
Real A = lambda.at<Real>(0), a = lambda.at<Real>(1), b = lambda.at<Real>(2);

    return A * exp((a - x) / b) / (b * pow(1. + exp((a - x) / b), 2.));

}//funcPrimeToFit

int makeFit(int dataSize, std::vector <datum>& data, cv::Mat& lambda) {
cv::Mat dfdl = (cv::Mat_<Real>(3, 1) << 0., 0., 0.);
cv::Mat beta = cv::Mat(dataSize, 1, CV_64FC1, 0.);
cv::Mat a = cv::Mat(dataSize, 3, CV_64FC1, 0.);

    for (int iter = 0; iter < 15; ++iter) {

        for (int i = 0; i < dataSize; ++i) {
            doDeriv(dfdl, data.at(i).X.x, lambda);
            for (int j = 0; j < 3; ++j) a.at<Real>(i, j) = dfdl.at<Real>(j);
        }//for_i

        for (int i = 0; i < dataSize; ++i) beta.at<Real>(i) = data.at(i).X.y - funcToFit(data.at(i).X.x, lambda);

        lambda += (a.t() * a).inv() * a.t() * beta; // delta lambda is added to lambda

    }//for_iter

    a.release();
    beta.release();
    dfdl.release();

    return 0;
}//makeFit

Real rsd(int dataSize, std::vector <datum>& data, cv::Mat& lambda) {

    // https://www.investopedia.com/terms/r/residual-standard-deviation.asp
    Real r2 = 0.;
    for (int i = 0; i < dataSize; ++i) {
        Real r = data.at(i).X.y - funcToFit(data.at(i).X.x, lambda);
        r2 += r * r;
    }//for_i

    return sqrt(r2 / (dataSize - 2));
}//rsd

int loadData(std::vector <saccade>& sLine, std::vector <datum>& data) {

    const Real Width = 1059., Height = 794., Px2Cm = 0.029461756;
    const Real X0 = sLine.at(SAC).fPoint.X.x * Width;
    const Real Y0 = sLine.at(SAC).fPoint.X.y * Height;
    const Real timeStamp0 = sLine.at(SAC).gPoint.at(0).timeStamp;

    for (int i = 0; i < (int)sLine.at(SAC).gPoint.size(); ++i) {

        Real X = sLine.at(SAC).gPoint.at(i).X.x * Width;
        Real Y = sLine.at(SAC).gPoint.at(i).X.y * Height;
        Real D = sqrt(pow(X - X0, 2.) + pow(Y - Y0, 2.)) * Px2Cm;
        Real theta = atan(D / 80.) * 180. / PI;
        Real t = sLine.at(SAC).gPoint.at(i).timeStamp - timeStamp0;
		datum blobT = (datum){ .X = {t, theta} };
		data.push_back(blobT);

    }//for_i

    Real X = sLine.at(SAC+1).fPoint.X.x * Width;
    Real Y = sLine.at(SAC+1).fPoint.X.y * Height;
    Real D = sqrt(pow(X - X0, 2.) + pow(Y - Y0, 2.)) * Px2Cm;
    Real theta = atan(D / 80.) * 180. / PI;
    Real t = sLine.at(SAC+1).fPoint.timeStamp - timeStamp0;
    datum blobT = (datum){ .X = {t, theta} };
    data.push_back(blobT);

    return 0;
}//loadData
/*
int readData(std::vector <datum> &dataT) {

	std::ifstream myFile("logistic12.dat");
	std::string line;

	while (getline(myFile, line)) {
		std::stringstream ss(line);
		Real x[2];
		ss >> x[0] >> x[1];
		datum blobT = (datum){ .X = {x[0], x[1]} };
		dataT.push_back(blobT);
	}//while

	myFile.close();

	return 0;
}//readData
*/
