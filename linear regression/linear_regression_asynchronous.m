
%setup
rho = 0;
p = 1000;
n = p/2;
nnzeros = 10; a = 2;

%data generating process
Sig=zeros(p,p);
 for i=1:p
    for k=1:p
         Sig(i,k)=rho^(abs(i-k));
    end
 end
 ch=chol(Sig,'lower');
 X=zeros(n,p);
 for i=1:n
     X(i,:)=(ch*randn(p,1))';
 end
for jj =1:p
    X(:,jj) = X(:,jj)*(sqrt(n)/norm(X(:,jj)));        
end

beta_true=zeros(p,1);
beta_idx = randsample(p,nnzeros);
for jj = 1:nnzeros
        %beta_true(ind_s(jj)) = 0;
        beta_true(beta_idx(jj))= (1-2*ceil(-1 + 2*rand(1)))*( a + rand(1));
end
y =  X*beta_true + randn(n,1);
testsize = 1000;
testX = randn(testsize,p);
for jj =1:p
    testX(:,jj) = testX(:,jj)*(sqrt(testsize)/norm(testX(:,jj)));        
end
testy =  testX*beta_true + randn(testsize,1);

%Exact    
Niter_Gibbs = 5*p;
display = 1;
rho_1 = 0.1;
rho_0 = n;
u = 0.5;
B = 100;
p_0= 100;
Asyn = 0;
full_data = 1;
Metrop_Step = 1;
gamma = 0.0005;
temp = 1;
thetahat = zeros(p,1);
thetahat(datasample(1:p,100)) = randn(100,1);
[Res_1,rr1,pmse1,pll_sample1] = approx_algo_lm(Niter_Gibbs, y, X,...
    display,rho_1,rho_0,u,B,p_0,Asyn,full_data,Metrop_Step,gamma,temp,beta_true, thetahat,testX,testy);

%Asyn
Niter_Gibbs = 5*p;
burnin = Niter_Gibbs -1000;
display = 1;
rho_1 = 0.1;
rho_0 = n;
u = 0.5;
B = 100;
p_0= 100;
Asyn = 1;
full_data = 1;
Metrop_Step = 1;
gamma = 0.0005;
temp = 1;
thetahat = zeros(p,1);
thetahat(datasample(1:p,100)) = randn(100,1);
[Res_2,rr2,pmse2,pll_sample2] = approx_algo_lm(Niter_Gibbs, y, X,...
    display,rho_1,rho_0,u,B,p_0,Asyn,full_data,Metrop_Step,gamma,temp,beta_true, thetahat,testX,testy);

function [Res,rr,pmse,pll_sample]  = approx_algo_lm(Niter, y, X, display,rho_1,rho_0,u,B,p_0,Asyn,full_data,Metrop_Step,gamma,temp, theta_star, thetahat,testX,testy)

%Gaussian-Gaussian spike-slab on sparsified linear regression model
% y, X are data
% options is  [rho, sigmasq,  display, gamma_0, u];
% theta_star is true values
%thetahat is initialization

    tic;

    %index of output
    delta_star = abs(theta_star) > 0;
    ind_pos = find(delta_star>0);
    norm_th = norm(theta_star);
    %data is n by p
    [n,p] = size(X);

    %recover inputted parameters
    rho_1_0 = rho_1;
    display = display;
    rho_0_0 = rho_0;
    u = u;
    B = B;
    p_0 = p_0;
    temp_vec = 1;
    nbTemp = 1;
    weights = zeros(1,nbTemp);
    step_weight = 10;
    mAcc_temp = 0;
    visits = zeros(1,nbTemp);
    var_eps = 1;
    a_uta = zeros(1,length(temp_vec));
    gamma = gamma;
    lgamma = log(gamma);
    Asynch = Asyn;
    full_data = full_data;
    Metrop_Step = Metrop_Step;
    const_q_0 = (u+1)*log(p) + 0.5*log(rho_0_0/rho_1_0);
    sigmasq_0 = 1;
    pmse = zeros(1,1000);


    %Initialization

    %Output
    Res=zeros(1,6);
    Res_full = zeros(1,p);
    count_temp_one = 0;


    %MCMC parameter needed
    delta  = abs(thetahat) > 0;    
    theta = thetahat;
    pll_sample =[];

    temp_ind = 1;

    for kk = 1:Niter
        temp = temp_vec(temp_ind);
        const_q = const_q_0/temp;
        rho_1 = rho_1_0/temp;
        rho_0 = rho_0_0;%/temp;
        sigmasq = sigmasq_0*temp;
        if full_data == 1
            mini_batch = 1:n;
            B = n;
        else
        mini_batch = datasample(1:1:n, B,'replace',false);
        end


        %Update delta
        coord_set = datasample(1:1:p, p_0, 'Replace', false);
        if Asynch == 0
            for jj = 1:p_0
                prob = const_q + 0.5*(rho_1-rho_0)*theta(coord_set(jj))^2; 
                delta_bar = delta; delta_bar(coord_set(jj)) = 1;
                prob = prob + (n/B)*(1/sigmasq)*0.5*(1/var_eps)*sum((y(mini_batch) - X(mini_batch,delta_bar)*theta(delta_bar)).^2);
                delta_bar = delta; delta_bar(coord_set(jj)) = 0;
                prob = prob - (n/B)*(1/sigmasq)*0.5*(1/var_eps)*sum((y(mini_batch) - X(mini_batch,delta_bar)*theta(delta_bar)).^2);
                prob = 1/(1 + exp(prob));
                delta(coord_set(jj)) = (rand(1)<=prob);
            end
        else
            coord_set_bar = [coord_set,setdiff(find(delta>0)',coord_set)];%find those delta =1 from coord_set 
            G = -rho_1*theta(coord_set_bar) ...
                + (n/B)*(1/sigmasq)* sum((1/var_eps)*(y(mini_batch) - X(mini_batch,coord_set_bar)*theta(coord_set_bar)).*X(mini_batch,coord_set_bar))';
            G = G(1:p_0);
            prob = 1./(1 +exp(const_q +0.5*(rho_1-rho_0)*theta(coord_set).^2 ...
                - theta(coord_set).*G));

            delta(coord_set) = (rand(p_0,1)<=prob);
        end
        %delta(coord_set) = temp_delta;


        %Update theta

        ind = find(delta==0);
        if isempty(ind)
        else
            theta(ind) = sqrt(1/rho_0)*randn(length(ind),1);
        end
        ind = find(delta==1);
        if isempty(ind)
        else
            if Metrop_Step == 1
                m = X(mini_batch,ind)'* X(mini_batch,ind) + rho_1*sigmasq*diag(length(ind));
                b = X(mini_batch,ind)'*y(mini_batch);
                mu = m^(-1)*b;
                theta(ind) = mu + (m/temp)^(-1/2)*randn(length(ind),1);
            else
                m = X(mini_batch,ind)*theta(ind);
                G = -rho_1*theta(ind) + (n/B)*(1/sigmasq)*sum((1/var_eps)*(X(mini_batch,ind).*(y(mini_batch) - m)))'; % gradient of conditional distribution of theta_1 given delta 
                theta_prop = theta(ind) + (gamma/2)*G + sqrt(gamma)*randn(length(ind),1);
                theta(ind) = theta_prop;
            end
        end 
        %temp_ind
        % Collecting relative error
        if temp_ind == 1
            count_temp_one = count_temp_one + 1;
            if norm_th == 0
                Res(count_temp_one,1) = norm(theta);
            else
                rel_norm = norm(theta.*delta-theta_star) / norm_th;
            end
            Res(count_temp_one,1) = rel_norm;
            Res_full(count_temp_one,:) = (theta.*delta);

            % ave. sensitivity and prec
            if isempty(ind_pos)
                sen = 1;
            else 
                sen = mean(delta(ind_pos)>0);
            end
            ind_pos_th = find(delta>0);
            if isempty(ind_pos_th)
                prec = 1;
            else
                prec = mean(delta_star(ind_pos_th)>0);
            end
            if (sen+prec) == 0
                Res(count_temp_one,2) = 0;
                Res(count_temp_one,3) = 0;
            else
                Res(count_temp_one,2) = sen;
                Res(count_temp_one,3) = prec;
            end
            Res(count_temp_one,4) = sum(delta);
            Res(count_temp_one,5) = mAcc_temp;
            Res(count_temp_one,6) = visits(1)/kk;
            if kk > p - 1000
                pmse(kk+1000- p) = norm(testy - testX*(theta.*delta))^2/1000;
            end    
            if display ~= 0
              [kk*1e-3, Res(count_temp_one,:)]
            end
        end

    end
    rr = norm(mean(Res_full(Niter-1000: Niter,:)) - theta_star')/norm_th;
end
