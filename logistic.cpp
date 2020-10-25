#include <fstream>
#include "auxiliary.hpp"

// https://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html

int main() {
int i, j;

	// Define a vector with initial size of zero elements and default fixation constructor
	std::vector <fixation> fPoint(0, (fixation){ .timeStamp = 0.f, .duration = 0.f, .X = {0.f, 0.f} }); // initial size, compound literal
	readFixations(fPoint);

	std::vector <gaze> gPoint(0, (gaze){ .timeStamp = 0.f, .X = {0.f, 0.f} });
	readGazes(gPoint);

//    std::cout << "Fixations: " << fPoint.size() << "\tGaze points: " << gPoint.size() << std::endl;

    std::vector <saccade> sLine(0, (saccade){ .fPoint = (fixation){ .timeStamp = 0., .X = {0., 0.} } } );

    int fPointSize = (int)fPoint.size();
    int gPointSize = (int)gPoint.size();

    for (i = 0; i < fPointSize-1; ++i) {

        Real t0 = fPoint.at(i).timeStamp;
        Real dt = fPoint.at(i).duration;
        Real fix = t0 + dt;

        // Looking for GAZE time stamps bigger than fix and less than next FIXATION time stamp
        saccade blobS = (saccade){ .fPoint = (fixation){ .timeStamp = t0, .duration = dt, .X = {fPoint.at(i).X.x, fPoint.at(i).X.y} } }; // initialize by fixation
        for (j = 0; j < gPointSize; ++j) {
            if (fix < gPoint.at(j).timeStamp && gPoint.at(j).timeStamp < fPoint.at(i+1).timeStamp) {
                gaze blobG = (gaze){ .timeStamp = gPoint.at(j).timeStamp, .X = {gPoint.at(j).X.x, gPoint.at(j).X.y} };
                blobS.gPoint.push_back(blobG); // initialize by gaze points
            }//if
        }//for_j
        sLine.push_back(blobS);

    }//for_i

    fPoint.clear();
    gPoint.clear();

	std::vector <datum> data(0, (datum){ .X = {0.f, 0.f} });
    loadData(sLine, data); // get selected saccade from sLine to data
	int dataSize = (int)data.size();

    for (i = 0; i < (int)sLine.size(); ++i) sLine.at(i).gPoint.clear();
    sLine.clear();

    // Here goes the real thing.
    cv::Mat lambda = (cv::Mat_<Real>(3, 1) << 0.);
/*
    // https://www.itl.nist.gov/div898/handbook/pmd/section6/pmd632.htm
    const Real gridY = 21;
    // A, t0, b
    cv::Mat c = (cv::Mat_<Real>(3, gridY) << 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5,
                                        0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105,
                                        0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021);
*/

    const Real gridY = 8;
    cv::Mat c = (cv::Mat_<Real>(3, gridY) << 3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3,
                                        0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07,
                                        0., 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035);

    Real r = 0., r0 = 100.;
    int c0, c1, c2;
    cv::Mat lambdaT = lambda.clone();
    // Find out the least Residual Standard Deviation!
    for (c0 = 0; c0 < gridY; ++c0)
        for (c1 = 0; c1 < gridY; ++c1)
            for (c2 = 0; c2 < gridY; ++c2) {

                lambda.at<Real>(0) = c.at<Real>(0, c0);
                lambda.at<Real>(1) = c.at<Real>(1, c1);
                lambda.at<Real>(2) = c.at<Real>(2, c2);

                makeFit(dataSize, data, lambda); // Solve for lambda
                r = rsd(dataSize, data, lambda);

                if (r < r0) {
                    r0 = r;
                    lambdaT.at<Real>(0) = lambda.at<Real>(0);
                    lambdaT.at<Real>(1) = lambda.at<Real>(1);
                    lambdaT.at<Real>(2) = lambda.at<Real>(2);
                }//if

            }//for_c2

    lambda.at<Real>(0) = lambdaT.at<Real>(0);
    lambda.at<Real>(1) = lambdaT.at<Real>(1);
    lambda.at<Real>(2) = lambdaT.at<Real>(2);
    lambdaT.release();

    std::cout << lambda << std::endl;

    cv::Mat img = imread("fw190tags.jpg", cv::IMREAD_COLOR);

    if(img.empty()) {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }//if
/*
std::cout << "Width : " << img.size().width << std::endl;
std::cout << "Height: " << img.size().height << std::endl;
*/

    std::vector<cv::Point2f> dataPoints;

    for (i = 0; i < (int)data.size(); ++i) {

        cv::Point2f new_point = cv::Point2f(5000. * data.at(i).X.x + 100., (Real)img.size().height - 100. * data.at(i).X.y);
        dataPoints.push_back(new_point);
        cv::circle(img, new_point, 2, CV_RGB(0, 0, 255), cv::FILLED, cv::LINE_AA, 0);
    }//for_i

//    https://answers.opencv.org/question/75445/how-to-draw-the-curve-line/
    std::vector<cv::Point2f> curvePoints;

    for (Real _x = 0.; _x < 0.05; _x += 0.001) {

        Real x = 5000. * _x + 100.;
        Real y = img.size().height - 100. * funcToFit(_x, lambda);
        cv::Point2f new_point = cv::Point2f(x, y);
        curvePoints.push_back(new_point);

    }//for_x

    cv::Mat curve(curvePoints, true);
    curve.convertTo(curve, CV_32S); //adapt type for polylines

    cv::polylines(img, curve, false, CV_RGB(255, 0, 0), 2, cv::LINE_AA);
    curve.release();

    // Compute local extreme value
    Real peakVel = 0., yMax0 = 0., atT = 0.;
    for (Real _x = 0.; _x < 0.04; _x += 0.001) {

        yMax0 = funcPrimeToFit(_x, lambda);
        if (yMax0 > peakVel) { peakVel = yMax0; atT = _x; }

    }//for_x

    Real amplitude = data.at(data.size()-1).X.y - data.at(0).X.y;
    Real duration  = data.at(data.size()-1).X.x - data.at(0).X.x;

    std::cout << atT << '\t' << peakVel << '\t' << amplitude << '\t' << duration << std::endl;
//    for (i = 0; i < (int)data.size(); ++i) std::cout << data.at(i).X << std::endl;

    imshow("Fitted curve", img);
    cv::waitKey(0); // Wait for a keystroke in the window

    c.release();
    data.clear();
    curvePoints.clear();
    dataPoints.clear();
    lambda.release();

	return 0;
}//main
