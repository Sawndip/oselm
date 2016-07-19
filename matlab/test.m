% test.m
if ~exist('loadMNIST', 'var') || ~loadMNIST
    run('/media/leoyolo/OS/Users/NTU/data/mnist/load_mnist_all.m');
end
rng('shuffle');
perm = randperm(size(xTrain, 1));
perm2 = randperm(size(xTest, 1), 1000);
initTrainSize = 6000;
xTrainSmall = xTrain(perm(1:initTrainSize), :);
yTrainSmall = yTrainExpanded(perm(1:initTrainSize), :);
xTestSmall = xTest(perm2, :);
yTestSmall = yTestExpanded(perm2, :);

numNeuron = 2000;
regConst = 100;

oselmClf = oselm(numNeuron);
oselmClf.init_train(xTrainSmall, yTrainSmall);
oselmClf.test(xTestSmall, yTestSmall);

batch_size = 100;
stats = [];
for i = 1:10
    i
    trainRange = perm(batch_size*(i-1)+initTrainSize:batch_size*i+initTrainSize);
    xTrainNew = xTrain(trainRange, :);
    yTrainNew = yTrainExpanded(trainRange, :);
    oselmClf.update(xTrainNew, yTrainNew);
    s = oselmClf.test(xTestSmall, yTestSmall);
    stats = [stats, s];
end
scores = oselmClf.compute_score(xTestSmall, false);
assert(all(size(scores) == size(yTestSmall)));
[~, predictedIds] = max(scores, [], 2);
[~, trueIds] = max(yTestSmall, [], 2);
acc = sum(predictedIds == trueIds)/numel(trueIds);
acc