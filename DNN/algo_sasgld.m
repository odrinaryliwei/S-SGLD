function [Res, Res_accur, dlnet]  = algo_sasgld(dlnet, Niter, YTrain, XTrain, YTest, XTest, opts)

    %data
    classes = unique(YTrain);% categories(YTrain);
    numClasses = numel(classes);
    numObservations = numel(YTrain);
    
    if opts.im_resize == 1
        XTest = imresize(XTest, [224,224]);
    end
    dlXTest = dlarray(XTest,'SSCB');
    
    if opts.gpu == 1
        dlXTest = gpuArray(dlXTest);
    end

    %recover inputted parameters
    num_class = 10;
    Monte_Carlo = opts.mc;
    initialLearnRate = opts.gam;
    decay = 0.001;
    miniBatchSize = opts.B;
    method = opts.method;
    tbSparse = opts.tbSparse; % speecify which layers should be sparsified
    rho_0_0 = opts.rho_0;
    rho_1_0 = opts.rho_1;
    u = opts.u;
    model_size = opts.model_size;
    p = model_size(1); p_0 = model_size(2);
    const_q_0 = (u+1)*log(p) + 0.5*log(rho_0_0/rho_1_0);
    temp_vec =opts.temp_vec;
    LL = size(dlnet.Learnables);
    param = dlnet.Learnables;
    param_struct = param;
    for ss = 1:LL(1)
        if opts.gpu == 1
            param_struct.Value{ss} = gpuArray(dlarray(ones(size(param.Value{ss}))));
        else
            param_struct.Value{ss} = dlarray(ones(size(param.Value{ss})));
        end
    end
    %gradients = param;
    
    %Output
    start = tic;
    Res=[]; Res_accur = []; Res_accur_rot = [];
    count_Update = 1;   

    plots = "training-progress";
    if plots == "training-progress"
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
        ylim([0 inf])
        xlabel("Iteration")
        ylabel("Loss")
        grid on
    end
    norm_lay = 0;
    temp_ind = 1;
    
    for kk = 1:Niter
        temp = temp_vec(temp_ind);
        const_q = const_q_0/temp;
        rho_1 = rho_1_0/temp;
        rho_0 = rho_0_0;
        scale = (1/temp)* numObservations;

        % Determine learning rate for time-based decay learning rate schedule.
        if Monte_Carlo == 1
            learnRate = temp*initialLearnRate/(1 + decay*kk^1);
        else
            learnRate = temp*initialLearnRate;
        end
        if method == "SGD"
            learnRate_sto = 0 ;
        else
            learnRate_sto = learnRate*2;
        end
        %Get minibatch
        idx = datasample(1:1:numObservations, miniBatchSize);
        X = XTrain(:,:,:,idx);
        if opts.im_resize == 1
            X = imresize(X, [224,224]);
        end

        Y = zeros(numClasses, miniBatchSize, 'single');
        for c = 1:numClasses
            Y(c,YTrain(idx)==classes(c)) = 1;
        end
        % Convert mini-batch of data to dlarray.
        dlX = dlarray(single(X),'SSCB');

        % If training on a GPU, then convert data to gpuArray.
        if opts.gpu == 1
            dlX = gpuArray(dlX);
        end
            
        nb_param = p_0;        
        %Update delta
        sparse_param = param;
        if sum(tbSparse) > 0
            coord_set = {};
            for ss = 1:LL(1)
                if tbSparse(ss) > 0
                    size_sel = numel(param.Value{ss});
                    ind_sel = datasample(1:1:size_sel, tbSparse(ss), 'Replace',false);
                    delta = param_struct.Value{ss};
                    delta(ind_sel) = 0;
                    sparse_param.Value{ss} = (param.Value{ss}).*delta;
                    coord_set.Value{ss} = ind_sel;
                end
            end
            dlnet.Learnables = sparse_param;
            [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
            dlnet.State = state;
            norm_lay = [];
            for ss = 1:LL(1)
                if tbSparse(ss) > 0
                    ind_sel = coord_set.Value{ss};
                    delta = param_struct.Value{ss};
                    grad = gradients.Value{ss};
                    grad = scale*reshape(grad(ind_sel),[1,length(ind_sel)]);
                    th = param.Value{ss};
                    th = reshape(th(ind_sel),[1,length(ind_sel)]);
                    prob = 1./(1 + exp(const_q + 0.5*(rho_1 - rho_0)*th...
                        -th.*grad -0.5*(th.^2).*(grad.^2)));
                    if isnan(extractdata(sum(prob)))
                        stop = 1;
                    end
                    if opts.gpu == 1
                        delta(ind_sel) = (rand(1,tbSparse(ss),'double','gpuArray') <= prob);
                    else
                        delta(ind_sel) = (rand(1,tbSparse(ss)) <= prob);
                    end
                    param_struct.Value{ss} = delta;
                    new_par = (param.Value{ss}).*delta;
                    sparse_param.Value{ss} = new_par;
                    nb_param = nb_param + sum(delta > 0, "all");
                end
            end
        end
        sparsity = nb_param / p;
        dlnet.Learnables = sparse_param;
        [gradients,state,loss] = dlfeval(@modelGradients,dlnet,dlX,Y);
        dlnet.State = state;
    
        % Update the network parameters
        %
        if method == "SGD"
            updateFun = @(param,grad) sgldUpdateFun(param,grad,learnRate,learnRate_sto,scale,rho_1);
            param = dlupdate(updateFun, param, gradients);
        elseif method == "SGLD"
            updateFun = @(param,grad) sgldUpdateFun(param,grad,learnRate,learnRate_sto,scale,rho_1);
            param = dlupdate(updateFun, param, gradients);
        elseif method == "SA-SGLD"
            updateFun = @(param,grad,param_struct) ssgldUpdateFun(param,grad,param_struct,...
                learnRate,learnRate_sto,scale,rho_1,rho_0);
            param = dlupdate(updateFun, param, gradients, param_struct);
        end
        
        
        %%Collect results
        
        if (temp_ind == 1)
            if length(temp_vec) > 1
                count_tmp = 100;
            else
                count_tmp = kk;
            end
            if mod(count_tmp,100) == 0
                % Display the training progress.
                if plots == "training-progress"
                    D = duration(0,0,toc(start),'Format','hh:mm:ss');
                    addpoints(lineLossTrain,kk,double(gather(extractdata(loss))))
                    drawnow
                end

                dlYPred = modelPredictions(dlnet, dlXTest, miniBatchSize, numClasses);
                [~,idx] = max(extractdata(dlYPred),[],1);
                YPred = classes(idx);
                accuracy = (YPred == YTest);               
                for gg = 1:num_class
                    Res_accur(count_Update,gg) = mean(accuracy(YTest==(gg-1)));
                end
                Res(count_Update,1) = mean(accuracy);
                Res(count_Update,2) = double(gather(extractdata(loss)));
                if sum(tbSparse) == 0
                    Res(count_Update,3) = 0;
                else
                    Res(count_Update,3) = double(gather(extractdata(sparsity)));
                end
                [kk, norm_lay, 100*Res(count_Update,:)]
                count_Update = count_Update + 1;
            end
        end
    end
    
end


function param = sgldUpdateFun(param, grad, lr, lr_sto, scale, rho_1)
    gradient = scale.*grad + rho_1.*param;
    param = param - lr.*gradient + sqrt(lr_sto).*randn(size(param));
end

function param = ssgldUpdateFun(param, grad, param_struct, lr, lr_sto, scale, rho_1, rho_0)
    gradient = scale.*grad + rho_1.*param;
    param = param - lr.*gradient + sqrt(lr_sto).*randn(size(param));
    param = param.*param_struct + (1-param_struct).*(1./sqrt(rho_0)).*randn(size(param));
end

function [gradients,state,loss] = modelGradients(dlnet,dlX,Y)

[dlYPred,state] = forward(dlnet,dlX);

loss = crossentropy(dlYPred,Y);
gradients = dlgradient(loss,dlnet.Learnables);

end

function dlYPred = modelPredictions(dlnet,dlX,miniBatchSize, numClasses)

numObservations = size(dlX,4);
numIterations = ceil(numObservations / miniBatchSize);

dlYPred = zeros(numClasses,numObservations,'like',dlX);

for i = 1:numIterations
    idx = (i-1)*miniBatchSize+1:min(i*miniBatchSize,numObservations);
    
    dlYPred(:,idx) = predict(dlnet,dlX(:,:,:,idx));
end

end

