#include "mnist.h"
#if defined(_MSC_VER)
#define byteswap32 _byteswap_ulong
#elif defined(__GNUC__) || defined(__GNUG__)
#define byteswap32 __builtin_bswap32
#endif
void mnist::load_images(const string &filename, int flag)
{
	assert(flag == MNIST_TRAIN || flag == MNIST_TEST);
	FILE *mnistImages = fopen(filename.c_str(), "rb");
	assert(mnistImages);
	int32_t magic=0;
	assert(sizeof(magic) == 4);
	int r0 = fread(&magic, sizeof magic, 1, mnistImages);
	assert(r0 == 1);
	magic = byteswap32(magic);	// This is the trick.
	assert(magic == 2051);

	int32_t numImages;
	int32_t numRows;
	int32_t numCols;
	int r1 = fread(&numImages, sizeof numImages, 1, mnistImages);	//TODO: error handling for whether reading is successful
	int r2 = fread(&numRows, sizeof numRows, 1, mnistImages);
	int r3 = fread(&numCols, sizeof numCols, 1, mnistImages);
	numImages = byteswap32(numImages);
	numRows = byteswap32(numRows);
	numCols = byteswap32(numCols);
	assert(r1 == 1);
	assert(r2 == 1);
	assert(r3 == 1);

	vector<uint8_t> images;
	images.resize(numImages*numRows*numCols);
	int r4 = fread(&images[0], sizeof images[0], images.size(), mnistImages);	//TODO: error handling
	assert(r4 == images.size());

	num_rows = numRows;
	num_cols = numCols;
	if (flag == MNIST_TRAIN)
	{
		_image_train = images;
		num_images_train = numImages;
	}
	else
	{
		if (flag == MNIST_TEST)
		{
			_image_test = images;
			num_images_test = numImages;
		}
	}
	fclose(mnistImages);
}

void mnist::load_labels(const string &filename, int flag)
{
	assert(flag == MNIST_TRAIN || flag == MNIST_TEST);
	//FILE *mnistLabels;
	FILE *mnistLabels = fopen(filename.c_str(), "rb");
	assert(mnistLabels);
	int32_t magic;
	fread(&magic, sizeof magic, 1, mnistLabels);
	magic = byteswap32(magic);
	assert(magic == 2049);
	int32_t numLabels;
	int r1 = fread(&numLabels, sizeof numLabels, 1, mnistLabels);
	numLabels = byteswap32(numLabels);
	assert(r1 == 1);

	vector<uint8_t> labels;
	labels.resize(numLabels);
	int r2 = fread(&labels[0], sizeof labels[0], labels.size(), mnistLabels);
	assert(r2 == labels.size());

	if (flag == MNIST_TRAIN)
	{
		_label_train = labels;
		num_labels_train = numLabels;
	}
	else
	{
		if (flag == MNIST_TEST)
		{
			_label_test = labels;
			num_labels_test = numLabels;
		}
		
	}
	fclose(mnistLabels);
}

void mnist::copy_to_mat(const vector<uint8_t>& vec, Mat& mat, const vector<int>& size)
// For the image size = {60000(10000), 28, 28}, and for the label size = {60000(10000)}.
{
	int innerSize = 1;
	if (size.size() == 1)
	{
		mat.create(size[0], 1, CV_32SC1);
		for (int i = 0; i < size[0]; ++i)
		{
			mat.ptr<int>(i)[0] = vec[i];
		}
	}
	else
	{
		for (int i = 1; i != size.size(); ++i)
			innerSize *= size[i];
		mat.create(size[0], innerSize, CV_64FC1);
		for (int i = 0; i < size[0]; ++i)
		{
			double *p = mat.ptr<double>(i);
			for (int j = 0; j < innerSize; ++j)
				p[j] = vec[i*innerSize + j] / 255.f;
		}
	}
	/*for (int i = 0; i != size[0]; ++i)
	{
		float *p = mat.ptr<float>(i);
		for (int j = 0; j != innerSize; ++j)
		{
			p[j] = vec[i*innerSize + j] / 255.f;
		}
	}*/
		//for (int j = 0; j != innerSize; ++j)
		//	mat.at<float>(i, j) = vec[i*innerSize + j] / 255.f;

}

void mnist::load_images_mat(const string& filename, int flag)
{
	this->load_images(filename, flag);
	vector<int> img_size;
	img_size.push_back(60000);
	img_size.push_back(28);
	img_size.push_back(28);
	if (flag == MNIST_TRAIN)
	{
		this->copy_to_mat(_image_train, image_train, img_size);
	}
	else
	{
		img_size[0] = 10000;
		this->copy_to_mat(_image_test, image_test, img_size);
	}
}

void mnist::load_labels_mat(const string& filename, int flag)
{
	this->load_labels(filename, flag);
	vector<int> label_size;
	label_size.push_back(60000);
	if (flag == MNIST_TRAIN)
	{
		this->copy_to_mat(_label_train, label_train, label_size);
	}
	else
	{
		label_size[0] = 10000;
		this->copy_to_mat(_label_test, label_test, label_size);
	}
}

void mnist::expand_labels(int flag)
{
	Mat label = flag == MNIST_TRAIN ? label_train : label_test;
	Mat labelExpanded = -Mat::ones(label.rows, 10, CV_64FC1);
	for (auto i = 0; i != label.rows; ++i)
	{
		auto idx = label.at<int>(i, 0);
		labelExpanded.at<double>(i, idx) = 1.;
	}
	if (MNIST_TRAIN == flag)
		label_train_expanded = labelExpanded;
	else
		label_test_expanded = labelExpanded;
}
