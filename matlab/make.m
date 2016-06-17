% build oselm
% EIGEN_DIR = C:\eigen-eigen-07105f7124f9\
if 1
    eigen_dir = 'C:\eigen-eigen-07105f7124f9\';
    mkl_dir = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2016.3.207\windows\mkl\';
    boost_dir = 'C:\boost_1_61_0\'
    mkl_include = fullfile(mkl_dir, 'include');
    mkl_lib = fullfile(mkl_dir, 'lib', 'intel64_win');
    openmp_lib = 'C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2016.3.207\windows\compiler\lib\intel64_win';
    boost_lib_vc14 = fullfile(boost_dir, 'lib64-msvc-14.0');
    boost_lib_vc12 = fullfile(boost_dir, 'lib64-msvc-12.0');
    eigen_switch = ['-I', eigen_dir];
    mkl_switch = ['-I', mkl_include];
    boost_switch = ['-I', boost_dir];
    mkl_lib_switch = ['-L', mkl_lib];
    openmp_switch = ['-L', openmp_lib];
    boost_lib_vc14_switch = ['-L', boost_lib_vc14];
    boost_lib_vc12_switch = ['-L', boost_lib_vc12];
    mex('-v', '-I..', eigen_switch, mkl_switch, boost_switch, mkl_lib_switch, ...
        '-lmkl_intel_lp64.lib', '-lmkl_core.lib', '-lmkl_intel_thread.lib', ...
        openmp_switch, '-llibiomp5md.lib', ...
        boost_lib_vc14_switch, 'boost_filesystem-vc140-mt-1_61.lib', 'boost_filesystem-vc140-mt-gd-1_61.lib', ...
        boost_lib_vc12_switch, 'boost_filesystem-vc120-mt-1_61.lib', 'boost_filesystem-vc120-mt-gd-1_61.lib', ...
        '-largeArrayDims', '-DEIGEN_USE_MKL_ALL', 'oselm_mex.cpp');
else
    mex -I.. -IC:\eigen-eigen-07105f7124f9\ COMPFLAGS="/openmp $COMPFLAGS" -largeArrayDims oselm_mex.cpp
end