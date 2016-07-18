#ifndef __MNIST_H__
#define __MNIST_H__

// This is a declaration that the safe version of fopen is not used.
//#define _CRT_SECURE_NO_WARNINGS

//#include <stdio.h>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cassert>
//#include "opencv_config.h"
#include "opencv2/opencv.hpp"


using namespace std;
using namespace cv;

enum { MNIST_TRAIN = 0, MNIST_TEST = 1 };

class mnist
{

public:
	typedef unsigned char uint8_t;
	typedef int int32_t;

	Mat image_train;
	Mat image_test;
	Mat label_train;
	Mat label_test;
	Mat label_train_expanded;
	Mat label_test_expanded;

	int num_images_train;
	int num_images_test;
	int num_rows;
	int num_cols;
	int num_labels_train;
	int num_labels_test;

	void load_images(const string &filename, int flag);
	void load_labels(const string &filename, int flag);
	void copy_to_mat(const vector<uint8_t> &vec, Mat &mat, const vector<int>& size);

	void load_images_mat(const string &filename, int flag);
	void load_labels_mat(const string &filename, int flag);

	void expand_labels(int flag);

	vector<uint8_t> _image_train;
	vector<uint8_t> _image_test;
	vector<uint8_t> _label_train;
	vector<uint8_t> _label_test;
};




#endif // __MNIST_H__