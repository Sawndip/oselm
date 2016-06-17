#include "class_handle.hpp"
#include "oselm.h"
#include "mex.h"
#include <map>
#include <string>
#include <functional>

typedef std::map<std::string, 
        std::function<void(int, mxArray **, int, const mxArray **)>> registryT;
typedef oselm<double, true> oselmD;

inline void mxCheck(bool expr, const char *msg)
{
    if (!expr) mexErrMsgTxt(msg);
}

// Usage: oselmObj = oselm_mex("new", num_neuron[, regConst])
static void oselm_create(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nlhs != 1)
        mexErrMsgTxt("New: One output expected.");
    if (nrhs != 2 && nrhs != 3)
        mexErrMsgTxt("New: Input numNeuron (mandatory) and regConst (optional).");
    double numNeuron = mxGetScalar(prhs[1]);
    double regConst = 0;
    if (nrhs == 3)
        regConst = mxGetScalar(prhs[2]);
    plhs[0] = convertPtr2Mat<oselmD>(new oselmD((int)numNeuron, regConst));
    return;
}
// Uasge: oselm_mex("delete", oselmObj);
static void oselm_delete(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 2)
        mexErrMsgTxt("Delete: Input the object to delete.");
    destroyObject<oselmD>(prhs[1]);
    return;
}
// Usage: oselm_mex("init_train", oselmObj, xTrain, yTrain)
static void oselm_init_train(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 4)
        mexErrMsgTxt("Usage: oselm_mex(\"init_train\", oselmObj, xTrain, yTrain)");
    oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
    double *xTrainPtr = mxGetPr(prhs[2]);
    size_t xrows = mxGetM(prhs[2]);
    size_t xcols = mxGetN(prhs[2]);
    double *yTrainPtr = mxGetPr(prhs[3]);
    size_t yrows = mxGetM(prhs[3]);
    size_t ycols = mxGetN(prhs[3]);
    int flag = oselmClassifier->oselm_init_train(xTrainPtr, (int)xrows, (int)xcols, 
            yTrainPtr, (int)yrows, (int)ycols);
    mxCheck(flag == 0, "Initialization is not successful.");
    return;
}
// Usage: oselm_mex("update", oselmObj, xTrain, yTrain)
static void oselm_update(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 4)
        mexErrMsgTxt("Usage: oselm_mex(\"update\", oselmObj, xTrain, yTrain)");
    oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
    double *xTrainPtr = mxGetPr(prhs[2]);
    size_t xrows = mxGetM(prhs[2]);
    size_t xcols = mxGetN(prhs[2]);
    double *yTrainPtr = mxGetPr(prhs[3]);
    size_t yrows = mxGetM(prhs[3]);
    size_t ycols = mxGetN(prhs[3]);
    mxCheck(xrows == yrows, "Number of samples in X and Y do not align.");
    mxCheck(xcols == oselmClassifier->get_feature_length(), 
            "Size of X does not align with feature length.");
    mxCheck(ycols == oselmClassifier->get_num_classes(),
            "Size of Y does not align with number of classes.");
    int flag = oselmClassifier->update(xTrainPtr, yTrainPtr, (int)xrows);
    mxCheck(flag == 0, "Update is not successful.");
    return;
}
// Usage: oselm_mex("compute_score", oselmObj, xTrain)
static void oselm_compute_score(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
	if (nrhs != 3)
		mexErrMsgTxt("Usage: oselm_mex(\"compute_score\", oselmObj, xTrain)");
	oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
	double *xTrainPtr = mxGetPr(prhs[2]);
	size_t xrows = mxGetM(prhs[2]);
	size_t xcols = mxGetN(prhs[2]);
	mxCheck(xcols == oselmClassifier->get_feature_length(),
			"Size of X does not align with feature length.");
	if (nlhs > 0)
	{
		//plhs[0] = mxCreateDoubleMatrix((int)xrows, oselmClassifier->get_num_classes(), mxREAL);
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		auto M = static_cast<size_t>(xrows);
		auto N = static_cast<size_t>(oselmClassifier->get_num_classes());
		mxSetM(plhs[0], M);
		mxSetN(plhs[0], N);
		mxSetData(plhs[0], mxMalloc(sizeof(double)*M*N)); // This is more efficient in allocating memory.
		double *scoresPtr = mxGetPr(plhs[0]);
		oselmClassifier->compute_score(scoresPtr, xTrainPtr, (int)xrows, (int)xcols, true);
	}
	return;
}

// Usage: oselm_mex("test", oselmObj, xTest, yTest)
static void oselm_test(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 4 && nrhs != 5)
        mexErrMsgTxt("Usage: oselm_mex(\"test\", oselmObj, xTest, yTest[, threshold])");
    oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
    double *xTestPtr = mxGetPr(prhs[2]);
    size_t xrows = mxGetM(prhs[2]);
    size_t xcols = mxGetN(prhs[2]);
    double *yTestPtr = mxGetPr(prhs[3]);
    size_t yrows = mxGetM(prhs[3]);
    size_t ycols = mxGetN(prhs[3]);
    double threshold = 0;
    if (nrhs == 5)
    {
        threshold = mxGetScalar(prhs[4]);
        if (oselmClassifier->get_num_classes() == 1)
            mexWarnMsgTxt("Input threshold is redundant and is not used.");
    }
    vector<double> stats = oselmClassifier->oselm_test(xTestPtr, (int)xrows, (int)xcols,
            yTestPtr, (int)yrows, (int)ycols, threshold);
    if (nlhs > 0)
    {
        plhs[0] = mxCreateDoubleMatrix((int)stats.size(), 1, mxREAL);
        double *statsPtr = mxGetPr(plhs[0]);
        for (int i = 0; i < stats.size(); ++i)
            statsPtr[i] = stats[i];
    }
    return;
}
// Usage: oselm_mex("snapshot", oselmObj, filename)
static void oselm_snapshot(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 3)
        mexErrMsgTxt("Usage: oselm_mex(\"snapshot\", oselmObj, filename");
    oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
    // char str[256];
    // mxCheck(mxGetM(prhs[2]) == 1, "The input filename should be a row vector of chars.");
    // int mxGetString(prhs[2], str, mxGetM(prhs[2])*mxGetN(prhs[2])+1);
    oselmClassifier->snapshot(std::string(mxArrayToString(prhs[2])));
    return;
}
// Usage: oselm_mex("load_snapshot", oselmObj, filename)
static void oselm_load_snapshot(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if (nrhs != 3)
        mexErrMsgTxt("Usage: oselm_mex(\"load_snapshot\", oselmObj, filename");
    oselmD *oselmClassifier = convertMat2Ptr<oselmD>(prhs[1]);
    // char str[256];
    // mxCheck(mxGetM(prhs[2]) == 1, "The input filename should be a row vector of chars.");
    // int mxGetString(prhs[2], str, mxGetM(prhs[2])*mxGetN(prhs[2])+1);
    oselmClassifier->load_snapshot(std::string(mxArrayToString(prhs[2])));
    return;
}

static registryT handlers {
    {"new", oselm_create},
    {"delete", oselm_delete},
    {"init_train", oselm_init_train},
    {"update", oselm_update},
    {"test", oselm_test},
    {"compute_score", oselm_compute_score},
    {"snapshot", oselm_snapshot},
    {"load_snapshot", oselm_load_snapshot}
};

void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    if(nrhs < 1)
        mexErrMsgTxt("Usage: oselm_mex(command, arg1, arg2, ...)");
    // Note that this actually brings memory leaks in mxArrayToString, referring to
    // http://www.mathworks.com/help/releases/R2016a/matlab/apiref/mxarraytostring.html
    std::string cmd(mxArrayToString(prhs[0]));
    auto search = handlers.find(cmd);
    if (handlers.end() == search)
        mexErrMsgTxt("Cannot find the specified command.");
    else
        search->second(nlhs, plhs, nrhs, prhs);
    return;
}