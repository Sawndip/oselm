#ifndef __MNIST_PATH_H__
#define __MNIST_PATH_H__

#include <string>
using std::string;

const string mnist_path = "/media/leoyolo/OS/Users/NTU/data/mnist/";
static string IMAGE_TRAIN = mnist_path + "train-images.idx3-ubyte";
static string IMAGE_TEST  = mnist_path + "t10k-images.idx3-ubyte";
static string LABEL_TRAIN = mnist_path + "train-labels.idx1-ubyte";
static string LABEL_TEST  = mnist_path + "t10k-labels.idx1-ubyte";
//const string IMAGE_TRAIN = "C:\\Users\\NTU\\data\\mnist\\train-images.idx3-ubyte";
//const string IMAGE_TEST = "C:\\Users\\NTU\\data\\mnist\\t10k-images.idx3-ubyte";
//const string LABEL_TRAIN = "C:\\Users\\NTU\\data\\mnist\\train-labels.idx1-ubyte";
//const string LABEL_TEST = "C:\\Users\\NTU\\data\\mnist\\t10k-labels.idx1-ubyte";




#endif // __MNIST_PATH_H__
