% build oselm
% EIGEN_DIR = C:\eigen-eigen-07105f7124f9\
mex -I.. -IC:\eigen-eigen-07105f7124f9\ COMPFLAGS="/openmp $COMPFLAGS" -largeArrayDims oselm_mex.cpp