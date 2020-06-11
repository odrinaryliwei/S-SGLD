function [Res, Rsq, pvar, pmse]  = algo_lenet5MNIST(net, Niter, y, X, y_test, X_test, opts)

    %Approx Gaussian-Gaussian spike-slab on CNN
    %The structure of the CNN is in net
    % y, X are training data; y_test, X_test are testing data
    % options is  [rho_0, rho_1,  display, gamma_0, u, B, NS];
    %Only fully connected layers are sparsified

    %data
    [dim_x, dim_y, nch, n] = size(X);
    [dim_x, dim_y, nch, n_test] = size(X_test);
    y_test = y_test + 1;
    y = y + 1;
    
    %recover inputted parameters
    display = opts.display;
    rho_1_0 = opts.rho_1;
    rho_0_0 = opts.rho_0;
    gamma = opts.gamma;
    u = opts.u;
    B = opts.B;
    NS = opts.NS;
    num_class = opts.num_class;
    

    %setup the network parameters
    depth_conv = 2; depth_full = 3;
    p = 0;     %total number of parameters
    Weights = {}; Bias = {}; dWeights={}; dBias={};
    for jj = 1:depth_conv
        conv = net{2*(jj-1)+1, 2};
        Weights(jj) = {dlarray(randn(conv(1),conv(1),conv(2), conv(5)))};
        Bias(jj) = {dlarray(randn(conv(5),1))};
        p = p + conv(1)*conv(1)*conv(2)*conv(5);
        p = p + conv(5); 
    end
    p_conv = p;
    
    for jj = 1:depth_full
        if jj == 1
            input = net{2*depth_conv + 1,2};
        else
            input = output;
        end
        output = net{2*depth_conv + jj+1, 2};
        Weights(depth_conv + jj) = {dlarray(randn(output, input))};
        Bias(depth_conv + jj) = {dlarray(randn(output,1))};
        dWeights(jj) = {zeros(output, input)};
        dBias(jj) = {zeros(output,1)};
        p = p + output*input;
        p = p + output; %calculating total number of parameters in Lenet-5
    end
    params.Weights = Weights; params.Bias = Bias;
    params_struct.Weights = dWeights; % whether parameter element (in full layer) is 0 or not
    params_struct.Bias = dBias;
    calc_grad = 1;

    %prior
    const_q_0 = (u+1)*log(p) + 0.5*log(rho_0_0/rho_1_0);
    temp = 1;
    rho_1 = rho_1_0/temp;
    rho_0 = rho_0_0/temp;
    scale = (1/temp)*(n/B);
    const_q = const_q_0/temp;
    
    %Output
    Res=zeros(1,num_class + 2);
    Rsq = zeros(400,length(y_test)); %R^2
    pvar =zeros(400,length(y_test)); %posterior variance
    pmse = zeros(400,length(y_test));%posterior mse
    pred_m1 = zeros(num_class, length(y_test)); 
    mse = zeros(1,length(y_test));
    count_est = 1;
    y_test_mat = zeros(num_class,length(y_test));
    y_test_mat((0:(length(y_test)-1))*num_class + y_test) = 1;% 10*ytest matrix
    

    for kk = 1:Niter

        mini_batch = datasample(1:1:n, B); % select a mini-batch of size B
        mini_dataX = X(:,:,:,mini_batch); 
        mini_dataY = y(mini_batch);

        %Update delta
        coord_set = {}; tmp_params_struct = params_struct;
        for jj = 1:depth_full
            size_sel = numel(params.Weights{depth_conv+jj});
            ind_sel = datasample(1:1:size_sel, NS(2), 'Replace', false);%update 20 weights per iteration
            delta = params_struct.Weights{jj};
            delta(ind_sel) = 1;
            tmp_params_struct.Weights(jj) = {delta};
            coord_set.Weights(jj) = {ind_sel};
            
            size_sel = numel(params.Bias{depth_conv+jj});
            ind_sel = datasample(1:1:size_sel, NS(1), 'Replace', false);%update 5 bias per iteration
            delta = params_struct.Bias{jj};
            delta(ind_sel) = 1;
            tmp_params_struct.Bias(jj) = {delta};
            coord_set.Bias(jj) = {ind_sel};
        end

        tmp_params = Sparsify(params, tmp_params_struct,depth_conv);
        [fval, gradients] = dlfeval(@model, net, num_class, mini_dataX, mini_dataY, calc_grad, tmp_params);
        norm_delta = 0;
        for jj = 1:depth_full
            SS = NS(2);
            delta = params_struct.Weights{jj};
            GW = extractdata(gradients.Weights{depth_conv + jj});
            W = extractdata(params.Weights{depth_conv + jj});
            grad = scale*GW(coord_set.Weights{jj});
            prob = 1./(1 +exp(const_q +0.5*(rho_1-rho_0)*W(coord_set.Weights{jj}).^2 ...
                - W(coord_set.Weights{jj}).*grad)); % calculate q_j w.r.t. weight parameters
            delta(coord_set.Weights{jj}) = (rand(1,SS)<= prob);
            params_struct.Weights(jj) = {delta};
            
            SS = NS(1);
            delta = params_struct.Bias{jj};
            GW = extractdata(gradients.Bias{depth_conv + jj});
            W = extractdata(params.Bias{depth_conv + jj});
            grad = scale*GW(coord_set.Bias{jj});
            prob = 1./(1 +exp(const_q +0.5*(rho_1-rho_0)*W(coord_set.Bias{jj}).^2 ...
                - W(coord_set.Bias{jj}).*grad));  % calculate q_j w.r.t. bias parameters
            delta(coord_set.Bias{jj}) = (rand(SS,1)<= prob);
            params_struct.Bias(jj) = {delta};
            
            norm_delta = norm_delta + sum(delta,'all');
        end
        
        tmp_params = Sparsify(params, params_struct,depth_conv);
        [fval, gradients] = dlfeval(@model, net, num_class, mini_dataX, mini_dataY, calc_grad, tmp_params);
        
        %Update parameters
        
        %SGLD on Convolutional layers
        for jj = 1:depth_conv
            th = extractdata(params.Weights{jj});
            G = extractdata(gradients.Weights{jj});
            grad = scale*G-rho_1*th;
            th = th + 0.5*gamma*grad +sqrt(gamma)*randn(size(th));
            params.Weights(jj) = {dlarray(th)};
            
            th = extractdata(params.Bias{jj});
            G = extractdata(gradients.Bias{jj});
            grad = scale*G-rho_1*th;
            th = th + 0.5*gamma*grad +sqrt(gamma)*randn(length(th),1);
            params.Bias(jj) = {dlarray(th)};
        end
        
        % update theta in dense layers
        for jj = 1:depth_full
            delta = params_struct.Weights{jj};
            th = extractdata(params.Weights{depth_conv+jj});
            ind = find(delta==0);
            if length(ind) > 0
                th(ind) = sqrt(1/rho_0)*randn(length(ind),1); % sample theta from N(0,1/rho_0) if delta = 0 
            end
            ind = find(delta==1);
            if length(ind) > 0 
                G = extractdata(gradients.Weights{depth_conv+jj});
                grad = scale*G(ind) - rho_1*th(ind);
                th(ind) = th(ind) + 0.5*gamma*grad + sqrt(gamma)*randn(length(ind),1);%SGLD update if delta = 1
            end
            params.Weights(depth_conv+jj) = {dlarray(th)};

            delta = params_struct.Bias{jj};
            th = extractdata(params.Bias{depth_conv+jj});
            ind = find(delta==0);
            if length(ind) > 0
                th(ind) = sqrt(1/rho_0)*randn(length(ind),1);
            end
            ind = find(delta==1);
            if length(ind) > 0 
                G = extractdata(gradients.Bias{depth_conv+jj});
                grad = scale*G(ind) - rho_1*th(ind);
                th(ind) = th(ind) + 0.5*gamma*grad + sqrt(gamma)*randn(length(ind),1);
            end
            params.Bias(depth_conv+jj) = {dlarray(th)};

        end

        % Collecting results
        if (kk>15000) && (mod(kk,50) == 0)% 15000 is the burn-in stage, this may vary for different datasets
            tmp_params = Sparsify(params, params_struct ,depth_conv);
            pred = model_prediction(net, X_test,  tmp_params);
            mse = mse + (1/count_est)*(sum( (pred - y_test_mat).^2 ) - mse);%taking average of mse among count_est steps
            pred_m1 = pred_m1 + (1/count_est)*(pred - pred_m1);
            count_est = count_est + 1;
            var_m = sum(pred_m1 - pred_m1.^2);
            pmse(count_est,:) = mse; 
            pvar(count_est,:) = var_m;
            Rsq(count_est,:) = var_m ./ mse; % R^2 we defined in the main paper
            ind = (mse == 0) & (var_m ==0 );
            Rsq(count_est,ind) = 1;
            for gg =1:num_class
                Res(kk,gg) = sum(Rsq(count_est,y_test == gg)) / sum(y_test == gg);
            end
            Res(kk,num_class + 1) = sum(pred((0:(length(y_test)-1))*num_class + y_test))/length(y_test);
        end
        sp = p_conv;
        for jj =1:depth_full
            sp = sp + nnz(params_struct.Weights{jj});
            sp = sp + nnz(params_struct.Bias{jj});
        end
        Res(kk,num_class + 2) = sp/p; % sp is the sparsity
                
        if display ~= 0
          [kk*1e-4, Res(kk,:)]
        end
    end
end

function [logpost, grad] = model(net, num_class, dataX, dataY, calc_grad, params)
% Calculate sparse log-posterior and its gradient for the model spec by
% net
    batch_size = size(dataX, 4);
    depth_conv = 2; depth_full = 3;
    dlz = dataX;
    for jj =1:depth_conv
        conv = net{2*(jj-1)+1,2};
        th = params.Weights{jj}; b = params.Bias{jj};
        dlz = dlconv(dlz, th, b, 'DataFormat','SSCB', 'stride', conv(3),'padding',conv(4));
%         offset = zeros(conv(5),1); scaleFactor = ones(conv(5),1);        
%         [dlz, mu, sigma] = batchnorm(dlz,offset,scaleFactor,'Dataformat','SSCB');
        dlz = relu(dlz); %or use relu
        pooling = net{2*jj,2};
        if gpuDeviceCount > 0
            dlz = gpuArray(dlz);
        end
        dlz = maxpool(dlz, pooling(1), 'DataFormat','SSCB', 'stride', pooling(2),'padding',pooling(3));
    end
    flat = net{2*depth_conv+1,2};
    dlz = reshape(dlz,flat,batch_size);
    for jj =1:depth_full
        W = params.Weights{depth_conv + jj};
        nnzW = find(extractdata(sum(W~=0))>0);% find columns of W where entries are not all 0;reduce calculation
        W = W(:,nnzW); 
        b = params.Bias{depth_conv + jj};
        dlz = b + W*dlz(nnzW,:);
        dlz = relu(dlz);
    end
    
    yind = (0:1:(batch_size-1))*num_class + dataY;
    max_z = max(dlz);
    z4 = dlz - repmat(max_z,num_class,1);
    logpost = sum(dlz(yind) - max_z) - sum(log(sum(exp(z4),1)));
    
%     yind = (0:1:(batch_size-1))*num_class + dataY;
%     zhat = dlz(yind);
%     z4 = dlz - repmat(zhat,num_class,1);
%     maxz4 = max(z4);
%     z5 = z4 - repmat(maxz4, num_class, 1);
%     logpost = -sum(maxz4 + log(sum(exp(z5),1)));

    if calc_grad == 0
        grad = 0;
    else
        % Automatic gradient
        grad = dlgradient(logpost, params, 'RetainData', true); 
    end
end

 
function tparams  = Sparsify(params, params_struct,depth_conv)
    %sparsify parameters
    tparams = params;
    depth = length(params.Weights);
    for jj = 1:(depth-depth_conv)
        tparams.Weights(depth_conv+jj) = {params.Weights{depth_conv+jj}.*params_struct.Weights{jj}};
        tparams.Bias(depth_conv+jj) = {params.Bias{depth_conv+jj}.*params_struct.Bias{jj}};        
    end
end


function pred = model_prediction(net, dataX,  params)
        % CNN prediction
        num_class = net{size(net,1),2};
        beta = 1;
        batch_size = size(dataX, 4);
        depth_conv = 2; depth_full = 3;
        dlz = dataX;
        for jj =1:depth_conv
            conv = net{2*(jj-1)+1,2};
            th = params.Weights{jj}; b = params.Bias{jj};
            dlz = dlconv(dlz, th, b, 'DataFormat','SSCB', 'stride', conv(3),'padding',conv(4));
%             offset = zeros(conv(5),1); scaleFactor = ones(conv(5),1);        
%             [dlz, mu, sigma] = batchnorm(dlz,offset,scaleFactor,'Dataformat','SSCB');
            dlz = relu(dlz); %or use sigmoid
            pooling = net{2*jj,2};
            dlz = maxpool(dlz, pooling(1), 'DataFormat','SSCB', 'stride', pooling(2),'padding',pooling(3));
        end
        flat = net{2*depth_conv+1,2};
        dlz = reshape(dlz,flat,batch_size);
        for jj =1:depth_full
            W = params.Weights{depth_conv + jj};
            nnzW = find(extractdata(sum(W~=0))>0);
            W = W(:,nnzW);
            b = params.Bias{depth_conv + jj};
            dlz = b + W*dlz(nnzW,:);
            dlz = relu(dlz);
        end
        
        dlz = extractdata(dlz);
        max_z = max(dlz);
        z4 = beta*(dlz - repmat(max_z,num_class,1));
        L = length(z4(1,:));
        z4 = exp(z4) ./ repmat(sum(exp(z4),1),num_class,1);
        pred = ( mnrnd(ones(L,1),z4') )';
        %find_ones = find(pr==1);
        %pred = mod(find_ones, ((0:(L-1))*num_class)')';
        
        %[~, pred] = max(dlz);
        
    end

