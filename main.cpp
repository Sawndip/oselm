#include "elm_base.h"
#include "mnist.h"
#include <iostream>
#include "mnist_path.h"
#include <numeric>
#include <algorithm>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "oselm.h"
#include <iterator>
#include <fstream>
using namespace cv;
using namespace std;
using namespace std::placeholders;	// for using std::bind
const int num_train = 2000;
const int num_test = 1000;
const int num_neuron = 5000;
const double elm_weight = 1e2;
const int num_mnist_train = 60000;
const int num_mnist_test = 10000;
const int batch_size = 100;
const int epoch = 20;

void test_elm()
{
	mnist mnist_loader;
	mnist_loader.load_images_mat(IMAGE_TRAIN, MNIST_TRAIN);
	mnist_loader.load_labels_mat(LABEL_TRAIN, MNIST_TRAIN);
	mnist_loader.load_images_mat(IMAGE_TEST, MNIST_TEST);
	mnist_loader.load_labels_mat(LABEL_TEST, MNIST_TEST);
	cout << "Successfully load MNIST dataset" << endl;

	std::vector<int> perm1(60000);
	std::vector<int> perm2(10000);
	std::iota(perm1.begin(), perm1.end(), 0);
	std::iota(perm2.begin(), perm2.end(), 0);
	srand(static_cast<unsigned>(clock()));
	std::random_shuffle(perm1.begin(), perm1.end());
	std::random_shuffle(perm2.begin(), perm2.end());

	Mat xTrain(num_train, 28 * 28, CV_64FC1);
	Mat yTrain(num_train, 10, CV_64FC1, Scalar(-1));
	Mat xTest(num_test, 28 * 28, CV_64FC1);
	Mat yTest(num_test, 10, CV_64FC1, Scalar(-1));

	for (auto i = 0; i < num_train; ++i)
	{
		auto id = perm1[i];
		auto pos = mnist_loader.label_train.at<int>(id, 0);
		yTrain.at<double>(i, pos) = 1.;
		mnist_loader.image_train.row(id).copyTo(xTrain.row(i));

	}
	for (auto i = 0; i < num_test; ++i)
	{
		auto id = perm2[i];
		auto pos = mnist_loader.label_test.at<int>(id, 0);
		yTest.at<double>(i, pos) = 1.;
		mnist_loader.image_test.row(id).copyTo(xTest.row(i));
	}
	xTrain = xTrain.clone();
	yTrain = yTrain.clone();
	xTest = xTest.clone();
	yTest = yTest.clone();
	cout << "Successfully create random samples from MNIST dataset" << endl;

	elm_base<double, false> elm_classifier(num_neuron, elm_weight);
	elm_classifier.elm_train((double *)xTrain.data, xTrain.rows, xTrain.cols, (double *)yTrain.data, yTrain.rows, yTrain.cols);
	elm_classifier.elm_test((double *)xTest.data, xTest.rows, xTest.cols, (double *)yTest.data, yTest.rows, yTest.cols);
}

void test_oselm()
{
	mnist mnist_loader;
	mnist_loader.load_images_mat(IMAGE_TRAIN, MNIST_TRAIN);
	mnist_loader.load_labels_mat(LABEL_TRAIN, MNIST_TRAIN);
	mnist_loader.load_images_mat(IMAGE_TEST, MNIST_TEST);
	mnist_loader.load_labels_mat(LABEL_TEST, MNIST_TEST);
	mnist_loader.expand_labels(MNIST_TRAIN);
	mnist_loader.expand_labels(MNIST_TEST);
	cout << "Successfully load MNIST dataset" << endl;

	Mat xTrain_init = mnist_loader.image_train(Range(0, num_train), Range::all()).clone();
	Mat yTrain_init = mnist_loader.label_train_expanded(Range(0, num_train), Range::all()).clone();
	Mat xTest_init = mnist_loader.image_test(Range(0, num_test), Range::all()).clone();
	Mat yTest_init = mnist_loader.label_test_expanded(Range(0, num_test), Range::all()).clone();

	// Construct consecutive range for update
	auto l = std::vector<int>(epoch);
	std::iota(l.begin(), l.end(), 0);
	auto range_train = std::vector<Range>();
	auto range_test = std::vector<Range>();
	auto trans_func = 
		[](int t, int init) -> cv::Range {return cv::Range(init + t*batch_size, init + (t+1)*batch_size);};
	std::transform(l.begin(), l.end(), std::back_inserter(range_train), std::bind(trans_func, _1, num_train));
	std::transform(l.begin(), l.end(), std::back_inserter(range_test), std::bind(trans_func, _1, num_test));
	//for (auto &r : range_train) cout << r.start << ", " << r.end << endl;
	//for (auto &r : range_test) cout << r.start << ", " << r.end << endl;

//	// set ostream for logging
//	std::ofstream of("logging.txt", std::ios::out);
//	CV_Assert(of.is_open() && of.good());

	oselm<double, false> oselm_classifier(num_neuron, elm_weight);
	oselm_classifier.oselm_init_train((double *)xTrain_init.data, xTrain_init.rows, xTrain_init.cols, 
		(double *)yTrain_init.data, yTrain_init.rows, yTrain_init.cols);
	oselm_classifier.oselm_test((double *)xTest_init.data, xTest_init.rows, xTest_init.cols, 
		(double *)yTest_init.data, yTest_init.rows, yTest_init.cols);
	oselm_classifier.get_stream() << "Testing for initializing oselm is successful." << endl;
	auto accuracy = std::vector<double>();
	for (auto e = 0; e != epoch; ++e)
	{
		Mat xTrain_new = mnist_loader.image_train(range_train[e], Range::all()).clone();
		Mat yTrain_new = mnist_loader.label_train_expanded(range_train[e], Range::all()).clone();
		Mat xTest_new = mnist_loader.image_test(range_test[e], Range::all()).clone();
		Mat yTest_new = mnist_loader.label_test_expanded(range_test[e], Range::all()).clone();
		oselm_classifier.get_stream() << "**Updating oselm on epoch " << e << endl;
		oselm_classifier.update((double *)xTrain_new.data, (double *)yTrain_new.data, batch_size);
		auto stats = oselm_classifier.oselm_test((double *)xTest_init.data, xTest_init.rows, xTest_init.cols,
			(double *)yTest_init.data, yTest_init.rows, yTest_init.cols);
		accuracy.push_back(stats.front());
	}
	oselm_classifier.get_stream() << "Testing for updating oselm is successful." << endl;
	oselm_classifier.get_stream() << "All accuracies in update process: ";
	for (auto &a : accuracy) oselm_classifier.get_stream() << a << "\t";
	cout << "\nTesting for oselm completes." << endl;
}


int main()
{
	//test_elm();
	test_oselm();
	return 0;
}