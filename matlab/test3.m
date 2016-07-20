if ~exist('loadMNIST', 'var') || ~loadMNIST
    run('/media/leoyolo/OS/Users/NTU/data/mnist/load_mnist_all.m');
end
rng('shuffle');

idTrain = find(yTrain == 0 | yTrain == 1);
idTest  = find(yTest  == 0 | yTest  == 1);

xTrainBinary = xTrain(idTrain, :);
yTrainBinary = yTrain(idTrain);
xTestBinary  = xTest(idTest, :);
yTestBinary  = yTest(idTest);

threshold = 0.5;
oselm_clf = oselm(2000, 100);
oselm_clf.train(xTrainBinary, yTrainBinary);
[acc, det, fa] = oselm_clf.test(xTestBinary, yTestBinary, threshold);