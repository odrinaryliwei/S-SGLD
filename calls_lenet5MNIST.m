%%If not already done, load and process MNIST data
%%use either MNIST or MNIST-FASHION for mnist or mnist-fashion
clear all
fashion = 0;

if fashion == 1    
    trainImagesFile = '../datasets/MNIST-FASHION/train-images-idx3-ubyte';
    trainLabelsFile = '../datasets/MNIST-FASHION/train-labels-idx1-ubyte';
    testImagesFile = '../datasets/MNIST-FASHION/t10k-images-idx3-ubyte';
    testLabelsFile = '../datasets/MNIST-FASHION/t10k-labels-idx1-ubyte';
    flattened = 0; %use flattened = 1 for mnist
else       
    trainImagesFile = '../datasets/MNIST/train-images-idx3-ubyte';
    trainLabelsFile = '../datasets/MNIST/train-labels-idx1-ubyte';
    testImagesFile = '../datasets/MNIST/t10k-images-idx3-ubyte';
    testLabelsFile = '../datasets/MNIST/t10k-labels-idx1-ubyte';
    flattened = 0; %use flattened = 1 for mnist
end
%}
XTrain = processMNISTimages(trainImagesFile, flattened);
YTrain = processMNISTlabels(trainLabelsFile);
numTrainImages = size(XTrain,4);
XTest = processMNISTimages(testImagesFile, flattened);
YTest = processMNISTlabels(testLabelsFile);
%angle = 180;
%XTest = imrotate(XTest,angle,'crop');
%sigma = 0;
%noise = sigma*randn(28,28,1);
%XTest = XTest + repmat(noise,1,1,1,10000);
num_class = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%Construct network structure
net_str = {};
depth_conv = 2; depth_full = 3;
%conv
padding = 1; stride = 1; w = 3; nch_in = size(XTrain,3); nch_out = 6;
net_str(1,1) = {'Conv'}; net_str(1,2) = {[w, nch_in, stride, padding, nch_out]};
input = [size(XTrain,1),size(XTrain,2),size(XTrain,3)];
output = [(input(1)+2*padding-w+stride)/stride, (input(2)+2*padding-w+stride)/stride,nch_out];
net_str(1,3) = {output};
%max pooling
padding = 0; stride = 2; w = 2;
net_str(2,1) = {'Max_Pool'}; net_str(2,2) = {[w, stride, padding]};
input = output;
output = [(input(1)+2*padding-w+stride)/stride, (input(2)+2*padding-w+stride)/stride,nch_out];
net_str(2,3) = {output};
%conv
padding = 1; stride = 1; w = 3; nch_in = nch_out; nch_out = 16;
net_str(3,1) = {'Conv'}; net_str(3,2) = {[w, nch_in, stride, padding, nch_out]};
input = output;
output = [(input(1)+2*padding-w+stride)/stride, (input(2)+2*padding-w+stride)/stride,nch_out];
net_str(3,3) = {output};
%max pooling
padding = 0; stride = 2; w = 2;
net_str(4,1) = {'Max_Pool'}; net_str(4,2) = {[w, stride, padding]};
input = output;
output = [(input(1)+2*padding-w+stride)/stride, (input(2)+2*padding-w+stride)/stride,nch_out];
net_str(4,3) = {output};

%Flattening
output = prod(output);
%output = 28*28;
net_str(2*depth_conv+1,1) = {'Flatten'}; net_str(2*depth_conv+1,2) ={output};
%full
net_str(2*depth_conv+2,1) = {'Full'}; net_str(2*depth_conv+2,2) = {300}; 
%full
net_str(2*depth_conv+3,1) = {'Full'}; net_str(2*depth_conv+3,2) = {100}; 
%full
net_str(2*depth_conv+4,1) = {'Full'}; net_str(2*depth_conv+4,2) = {num_class}; 


% % Approximate sampling
Niter = 30000;
options.display = 1;
options.rho_1 = 1;
options.rho_0 = 10;
options.B = 100;
options.NS= [5, 20];
options.gamma = 0.5*1e-5;
options.u = 1;
options.num_class = num_class;

tic
textout = 'Starting MCMC sampling.'
[Res, Rsq, pvar, pmse] = algo_lenet5MNIST(net_str, Niter, YTrain, XTrain, YTest, XTest, options);
time = toc;

Res = Res(Res(:,num_class+1)>0,:);

subplot(3,4,1); plot(Res(:,1),'-b'); 
subplot(3,4,2); plot(Res(:,2),'-b'); 
subplot(3,4,3); plot(Res(:,3),'-b');
subplot(3,4,4); plot(Res(:,4),'-b'); 
subplot(3,4,5); plot(Res(:,5),'-b'); 
subplot(3,4,6); plot(Res(:,6),'-b');
subplot(3,4,7); plot(Res(:,7),'-b'); 
subplot(3,4,8); plot(Res(:,8),'-b'); 
subplot(3,4,9); plot(Res(:,9),'-b');
subplot(3,4,10); plot(Res(:,10),'-b'); 
subplot(3,4,11); plot(Res(:,11),'-b'); 
subplot(3,4,12); plot(Res(:,12),'-b');
%%
if fashion == 1
    filename = ['MNIST_FASHION_APPROXIMATE_MC','.mat'];
else
    filename = ['MNIST_APPROXIMATE_MC','.mat'];
end
save(filename,'Res','Rsq','YTest','pvar','pmse')
