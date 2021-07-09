%%use either MNIST or MNIST-FASHION for mnist or mnist-fashion
%one could find mnist-fashion dataset from the following link
%https://github.com/zalandoresearch/fashion-mnist/tree/master/data
clear all
trainImagesFile = '../datasets/MNIST-FASHION/train-images-idx3-ubyte';
trainLabelsFile = '../datasets/MNIST-FASHION/train-labels-idx1-ubyte';
testImagesFile = '../datasets/MNIST-FASHION/t10k-images-idx3-ubyte';
testLabelsFile = '../datasets/MNIST-FASHION/t10k-labels-idx1-ubyte';

options.gpu = 1;%set to zero if do not apply gpu
flattened = 0; %use flattened = 1 for mnist
XTrain = processMNISTimages(trainImagesFile, flattened);
YTrain = processMNISTlabels(trainLabelsFile);
numTrainImages = size(XTrain,4);
XTest = processMNISTimages(testImagesFile, flattened);
XTest_rotat = XTest;
YTest = processMNISTlabels(testLabelsFile);
num_class = 10;

mean_img = mean(XTrain,4);
for kk = 1: length(YTrain)
    XTrain(:,:,:,kk)  = XTrain(:,:,:,kk) - mean_img;
end
mean_img = mean(XTest,4);
for kk = 1: length(YTest)
    XTest(:,:,:,kk)  = XTest(:,:,:,kk) - mean_img;
end


%Fit Lenet_5 --- Construct netwrok structure
layers_lenet = [
                  imageInputLayer([28 28 1], 'Name', 'input', 'Mean', mean(XTrain,4))
                  convolution2dLayer(3, 6, 'Stride', 1, 'Padding', 1, 'Name', 'conv1')
                  reluLayer('Name', 'relu1')
                  maxPooling2dLayer(2,'Stride', 2, 'Name', 'maxpool1')
                  convolution2dLayer(3, 16, 'Stride', 1, 'Padding', 1, 'Name', 'conv2')
                  reluLayer('Name', 'relu2')
                  maxPooling2dLayer(2,'Stride', 2, 'Name', 'maxpool2')
                  fullyConnectedLayer(300, 'Name', 'fc1')
                  fullyConnectedLayer(200, 'Name', 'fc2')
                  fullyConnectedLayer(num_class, 'Name', 'fc3')
                  softmaxLayer('Name','softmax')];
lgraph_lenet = layerGraph(layers_lenet);

dlnet_lenet = dlnetwork(lgraph_lenet);
net_depth = size(dlnet_lenet.Learnables);
Niter = 250e3;
options.B = 100;
options.gam = 1e-7;
options.rho_1 = 1;
options.rho_0 = 10000;
options.u = 50;
options.temp_vec = [1];
options.method = "SA-SGLD"; %use SGD to do stoch grad. descent.
options.mc = 0;
options.display = 1;
options.im_resize = 0;
%options.tbSparse= zeros(1,net_depth(1)); %use this to do pure SGLD
options.tbSparse = [10, 1, 10, 1, 300, 10, 200, 10, 10, 0]; % do pure SGLD is this vector is fully zero
p = 0; p_0 = 0;
for ss = 1:net_depth(1)
    p = p + numel(dlnet_lenet.Learnables.Value{ss});
    if options.tbSparse(ss) == 0
        p_0 = p_0 + numel(dlnet_lenet.Learnables.Value{ss});
    end
end
options.model_size = [p, p_0];

tic
    [Res_lenet, accuracy, dlnet_fitted] = algo_sasgld(dlnet_lenet, Niter, YTrain, XTrain, YTest, XTest, options);
time = toc;

save('lenet_mnist_fashion.mat');



