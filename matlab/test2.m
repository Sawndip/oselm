if ~exist('loadMNIST', 'var') || ~loadMNIST
    run('C:\Users\NTU\data\mnist\load_mnist_all.m');
end
rng('shuffle');
perm = randperm(size(xTrain, 1));
perm2 = randperm(size(xTest, 1), 100);
initTrainSize = 600;
xTrainSmall = xTrain(perm(1:initTrainSize), :);
yTrainSmall = yTrainExpanded(perm(1:initTrainSize), :);
xTestSmall = xTest(perm2, :);
yTestSmall = yTestExpanded(perm2, :);

numNeuron = 1000;
regConst = 100;

oselmClf = oselm(numNeuron, regConst);
oselmClf.init_train(xTrainSmall, yTrainSmall);
oselmClf.print_variables();
oselmClf.snapshot('iter2');
clear oselmClf;
oselmClf = oselm(numNeuron, regConst);
oselmClf.load_snapshot('iter2');
oselmClf.print_variables();
oselmClf.set_variables('num_classes', 10);
oselmClf.print_variables();
acc = oselmClf.test(xTestSmall, yTestSmall);
