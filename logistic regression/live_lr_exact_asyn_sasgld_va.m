%% 
% Set up & Data Generating Process

rho = 0;
p = 1000;
n = p;
n_test = 1000;
nnzeros = 10; a1 = 6;
%batch = 50;

%%------Generate data
   %Option 1
   Sig=zeros(p,p);
     for i=1:p
        for j=1:p
             Sig(i,j)=rho^(abs(i-j));
        end
     end
     ch=chol(Sig,'lower');
     X=zeros(n,p);
     for i=1:n
         X(i,:)=(ch*randn(p,1))';
     end
     X_test = zeros(n_test,p);
     for i=1:n_test
         X_test(i,:)=(ch*randn(p,1))';
     end
%X = randn(n,p);
%X(:,1:2) = mvnrnd([0,0],[1,0.9;0.9,1],n);
    %normalize column
    for jj =1:p
        X(:,jj) = X(:,jj)*(sqrt(n)/norm(X(:,jj)));        
    end
    
    for jj =1:p
        X_test(:,jj) = X_test(:,jj)*(sqrt(n_test)/norm(X_test(:,jj)));        
    end
    
    beta_true=zeros(p,1);

    for jj = 1:nnzeros
        beta_true(jj)= (1-2*ceil(-1 + 2*rand(1)))*( a1 + 0.1*rand(1));
    end
    
prob = 1./(1+ exp(-X*beta_true));
y =  (rand(n,1) <= prob);
prob_test = 1./(1+ exp(-X_test*beta_true));
y_test =  (rand(n_test,1) <= prob_test);

options.display = 0;% set to 1 if you want to observe the statistics per iteration
options.rho_1 = 0.1;
options.rho_0 = n;
options.u = 0.5;
options.B = 100;
options.p_0= 100;
options.temp_vec = 1;
%% 

% Lasso Initialization

[B,FitInfo] = lassoglm(X,y,'binomial','CV',10);
idxLambda1SE = FitInfo.Index1SE;
thetahat = B(:,idxLambda1SE);

%%
% Exact Algorithm:

%exact algorithm

Niter_Gibbs = 5*p;
options.Asyn = 0;
options.full_data = 1;
options.Metrop_Step = 1;
options.gamma = 5/n;

tic
[Res_1, Res_full_1,pll_1,pll_sample1] = algo_tempering_logistic(Niter_Gibbs, y, X,y_test,X_test,options, beta_true, thetahat);
rtime1 =toc;
rr_exact = mean(Res_1(Niter_Gibbs/50 - 20 : Niter_Gibbs/50,1));

%% 
% Next is Asyn Algorithm

%Asyn algorithm

Niter_Gibbs = 5*p;
options.Asyn = 1;
options.full_data = 1;
options.Metrop_Step = 1;
options.gamma = 5/n;

tic
[Res_2, Res_full_2,pll_2,pll_sample2] = algo_tempering_logistic(Niter_Gibbs, y, X,y_test,X_test,options, beta_true, thetahat);
rtime2 =toc;
rr_asyn = mean(Res_2(Niter_Gibbs/50 - 20 : Niter_Gibbs/50,1));
%%
% SA-SGLD

%SA-SGLD algorithm

Niter_Gibbs = 10*p;
options.Asyn = 1;
options.full_data = 0;
options.Metrop_Step = 0;
options.gamma = 1/n;

tic
[Res_3, Res_full_3,pll_3,pll_sample3] = algo_tempering_logistic(Niter_Gibbs, y, X,y_test,X_test,options, beta_true, thetahat);
rtime3 =toc;
rr_sasgld = mean(Res_3(Niter_Gibbs/50 - 20 : Niter_Gibbs/50,1));
%% 

%VA algorithm
Niter_Gibbs = 5*p;
options.rho_1 = 0.1;
options.rho_0 = 100;
options.display = 0;
mean_ = randn(p,1);
rho = zeros(p,1);
alpha_ = -3*ones(p,1);
tic
[diff,mu,niu,alpha] = New_ELBO_logistic(Niter_Gibbs, y, X, options, mean_,rho,alpha_,beta_true);
rr_VA = zeros(1000,1);  
for k = 1:1000
    delta = (rand(p,1) < 1./(1+exp(-alpha)));
    rr_VA(k) =  norm((mu + niu.*randn(p,1)).*delta - beta_true)/norm(beta_true);
end
vatime = toc;
rr_va = mean(rr_VA);
%%

% Present results 

% relative error of each algorithm
exact = [rr_exact,rtime1]';
asyn = [rr_asyn,rtime2]';
sasgld = [rr_sasgld,rtime3]';
va = [rr_va,vatime]';
out = [exact,asyn,sasgld,va];
%disp([exact,asyn,sasgld,va],  'RowNames', ...
%{'Relative Error', 'Runing time'});
table(out, 'VariableNames', {'mean'}, 'RowNames', {'Relative Error', 'Runing time'})
%,'VariableNames', {'Exact','Asyn','SA-SGLD','VA'}

%%
% Local functions:

function [Res, Res_full,pll_true,pll_sample]  = algo_tempering_logistic(Niter, y, X,y_test,X_test, opts, theta_star, thetahat)

%Gaussian-Gaussian spike-slab on sparsified logistic regression model
% y, X are data
% options is  [rho, sigmasq,  display, gamma_0, u];
% theta_star is true values
%thetahat is initialization

tic;

%index of output
delta_star = abs(theta_star) > 0;
ind_pos = find(delta_star>0);

%data is n by p
[n,p] = size(X);

%recover inputted parameters
rho_1_0 = opts.rho_1;
display = opts.display;
rho_0_0 = opts.rho_0;
u = opts.u;
B = opts.B;
p_0 = opts.p_0;
nbTemp = length(opts.temp_vec);
temp_vec = opts.temp_vec;
weights = zeros(1,nbTemp);
step_weight = 10;
mAcc_temp = 0;
visits = zeros(1,nbTemp);
a_uta =  zeros(1,nbTemp);

gamma = opts.gamma;
lgamma = log(gamma);
Asynch = opts.Asyn;
full_data = opts.full_data;
Metrop_Step = opts.Metrop_Step;

%prior
%prior_q = (1/p)^(u+1);
%const_q_0 = log(1-prior_q) -log(prior_q) + 0.5*log(rho_0_0/rho_1_0);
const_q_0 = (u+1)*log(p) + 0.5*log(rho_0_0/rho_1_0);
sigmasq_0 = 1;

pll_true = dot(y_test,X_test*theta_star) - sum(log(1 +exp(X_test*theta_star))) -0.5*rho_1_0*norm(theta_star)^2;
pll_sample = [];

%Initialization

for j = 1:nbTemp
    a_uta(j) = const_q_0 + (temp_vec(j) -1)* log(p);
end

%Output
Res=zeros(1,6);
Res_full = zeros(1,p);
count_temp_one = 0;


%MCMC parameter needed
delta  = abs(thetahat) > 0;    
theta = thetahat;

temp_ind = 1;


for kk = 1:Niter
    temp = temp_vec(temp_ind);
    const_q = const_q_0/temp;
    %const_q = (u+1)*log(p) + 0.5*(log(rho_0_0) - log(rho_1_0));
    rho_1 = rho_1_0/temp;
    rho_0 = rho_0_0;
    sigmasq = sigmasq_0*temp;

    if full_data == 1
        mini_batch = 1:n;
        B = n;
    else
        mini_batch = datasample(1:1:n, B);
    end
    
    %Update delta
    coord_set = datasample(1:1:p, p_0, 'Replace', false);
    if Asynch == 0
        for jj = 1:p_0
            prob = const_q + 0.5*(rho_1-rho_0)*theta(coord_set(jj))^2 ...%%A_j? 
                -(n/B)*(1/sigmasq)*theta(coord_set(jj))*dot(y(mini_batch),X(mini_batch,coord_set(jj)));
            delta_bar = delta; delta_bar(coord_set(jj)) = 1;
            prob = prob + (n/B)*(1/sigmasq)*sum(log(1 + exp(X(mini_batch,delta_bar)*theta(delta_bar))));
            delta_bar = delta; delta_bar(coord_set(jj)) = 0;
            prob = prob - (n/B)*(1/sigmasq)*sum(log(1 + exp(X(mini_batch,delta_bar)*theta(delta_bar))));
            prob = 1/(1 + exp(prob));
            delta(coord_set(jj)) = (rand(1)<=prob);
            %[prob, 0.5*(rho_1-rho_0)*theta(coord_set(jj))^2]
        end
    else
        coord_set_bar = [coord_set,setdiff(find(delta>0)',coord_set)];%find those delta =1 from coord_set 
        m = X(mini_batch,coord_set_bar)*theta(coord_set_bar);
        G = -rho_1*theta(coord_set_bar) ...
            + (n/B)*(1/sigmasq)*(X(mini_batch,coord_set_bar)')*(y(mini_batch) - exp(m)./(1 + exp(m)));
        G = G(1:p_0);
        %prob = 1./(1 +exp(const_q +0.5*(rho_1-rho_0)*theta(coord_set).^2 ...
         %   - theta(coord_set).*G ));
        prob = 1./(1 +exp(const_q +0.5*(rho_1-rho_0)*theta(coord_set).^2 ...
            - theta(coord_set).*G - 0.5*(theta(coord_set).^2).*(G.^2)));
        
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
        m = X(mini_batch,ind)*theta(ind);
        mp = y(mini_batch) - exp(m)./(1 + exp(m));
        G = -rho_1*theta(ind) + (n/B)*(1/sigmasq)*(X(mini_batch,ind)')*mp; % gradient of conditional distribution of theta_1 given delta 
        theta_prop = theta(ind) + (gamma/2)*G + sqrt(gamma)*randn(length(ind),1);
        
        if Metrop_Step == 1
            ll = -0.5*rho_1*norm(theta(ind))^2 + (1/sigmasq)*dot(y,X(:,ind)*theta(ind))...
                - (1/sigmasq)*sum(log(1 +exp(X(:,ind)*theta(ind))));
            ll_prop = -0.5*rho_1*norm(theta_prop)^2 + (1/sigmasq)*dot(y,X(:,ind)*theta_prop)...
                - (1/sigmasq)*sum(log(1 +exp(X(:,ind)*theta_prop)));
            m_prop = X(mini_batch,ind)*theta_prop;
            G_prop = -rho_1*theta_prop + (n/B)*(1/sigmasq)*(X(mini_batch,ind)')...
                *(y(mini_batch) - exp(m_prop)./(1 + exp(m_prop)));
            Acc = (ll_prop - ll) -(0.5/gamma)*(norm(theta(ind) - theta_prop -(gamma/2)*G_prop)^2 ...
                - norm(theta_prop - theta(ind) - (gamma/2)*G)^2);
            %[ll_prop-ll,Acc]
            if rand(1) <= min(1,exp(Acc))
                theta(ind) = theta_prop;
            end
            lgamma = lgamma + (1/kk^0.55)*(min(1,exp(Acc)) - 0.5);
            gamma = exp(lgamma);
        else
            theta(ind) = theta_prop;
        end
        %[kk,temp_ind]
    end
    
    %Update temperature
    if nbTemp > 1
        ind = find(delta==1);
        %U = -const_q_0*sum(delta) -0.5*rho_1_0*norm(theta.*delta)^2 ...
            %-0.5*rho_0_0*norm(theta-theta.*delta)^2;
        %U = U + (n/B)*dot(y(mini_batch),X(mini_batch,ind)*theta(ind)) ...
            %-(n/B)*sum(log(1 +exp(X(mini_batch,ind)*theta(ind))));%posterior dist(\theta|\delta,Z)
         U = (n/B)*dot(y(mini_batch),X(mini_batch,ind)*theta(ind)) ...
            -(n/B)*sum(log(1 +exp(X(mini_batch,ind)*theta(ind)))) -0.5*rho_1_0*norm(theta.*delta)^2;
        %U
        if (temp_ind == 1) 
            temp_ind_prop = 2;
            %const_q_prop = (u+1)*log(p) + 0.5*(temp_vec(temp_ind_prop)*log(rho_0_0) - log(rho_1_0));
            Acc_temp = weights(temp_ind) - weights(temp_ind_prop) ...
                - length(ind)*(a_uta(temp_ind_prop)/temp_vec(temp_ind_prop) - a_uta(temp_ind)/temp_vec(temp_ind)) ...
                + U*(1/temp_vec(temp_ind_prop) - 1/temp_vec(temp_ind));
            Acc_temp = min(1,0.5*exp(Acc_temp));
        elseif temp_ind == nbTemp
            temp_ind_prop = nbTemp-1;
            %const_q_prop = (u+1)*log(p) + 0.5*(temp_vec(temp_ind_prop)*log(rho_0_0) - log(rho_1_0));
            Acc_temp = weights(temp_ind) - weights(temp_ind_prop) ...
                - length(ind)*(a_uta(temp_ind_prop)/temp_vec(temp_ind_prop) - a_uta(temp_ind)/temp_vec(temp_ind)) ...
                + U*(1/temp_vec(temp_ind_prop) - 1/temp_vec(temp_ind));
            Acc_temp = min(1,0.5*exp(Acc_temp));
        else
            if rand(1) <= 0.5 
                temp_ind_prop = temp_ind - 1;
            else
                temp_ind_prop = temp_ind + 1;
            end
            %const_q_prop = (u+1)*log(p) + 0.5*(temp_vec(temp_ind_prop)*log(rho_0_0) - log(rho_1_0));
            Acc_temp = weights(temp_ind) - weights(temp_ind_prop) ...
                - length(ind)*(a_uta(temp_ind_prop)/temp_vec(temp_ind_prop) - a_uta(temp_ind)/temp_vec(temp_ind)) ...
                + U*(1/temp_vec(temp_ind_prop) - 1/temp_vec(temp_ind));
            if temp_ind_prop == 1
                Acc_temp = min(1,2*exp(Acc_temp));
            elseif temp_ind_prop == nbTemp
                Acc_temp = min(1,2*exp(Acc_temp));
            else
                Acc_temp = min(1,exp(Acc_temp));
            end
        end
        if rand(1) <= Acc_temp
            temp_ind = temp_ind_prop;
        end
        mAcc_temp = mAcc_temp + (1/kk)*(Acc_temp - mAcc_temp);
        weights(temp_ind) = weights(temp_ind) + step_weight;
        visits(temp_ind) = visits(temp_ind) + 1;
        if max(abs(visits/kk - 1/nbTemp)) < 0.5/nbTemp
            step_weight = step_weight/2;
            visits = zeros(1,nbTemp);
        end
    end
    %temp_ind
    % Collecting relative error
    if temp_ind == 1 && (mod(kk,50) == 0)
        count_temp_one = count_temp_one + 1;
        norm_th = norm(theta_star);
        if norm_th == 0
            Res(count_temp_one,1) = norm(theta);
        else
            rel_norm = norm(theta.*delta-theta_star) / norm_th;
        end
        Res(count_temp_one,1) = rel_norm;
        Res_full(count_temp_one,:) = theta';

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
        
        if display ~= 0
          [kk*1e-3, Res(count_temp_one,:)]
        end
        ind = find(delta==1);
        pll_sample(count_temp_one) = dot(y_test,X_test(:,ind)*theta(ind)) ...
            -sum(log(1 +exp(X_test(:,ind)*theta(ind)))) -0.5*rho_1_0*(norm(theta(ind))^2);
    end
end

%step_weight
end

function [diff,mu,niu,alpha] = New_ELBO_logistic(Niter, y, X, opts, mu, rho,alpha,beta_true)

%Gaussian-Gaussian spike-slab on sparsified logistic regression model
% y, X are data
% options is  [rho, sigmasq,  display, gamma_0, u];
% theta_star is true values
%thetahat is initialization

tic;

%index of output
%delta_star = abs(theta_star) > 0;
%ind_pos = find(delta_star>0);

%data is n by p
[n,p] = size(X);

%recover inputted parameters
%rho_1 = opts.rho_1;
display = opts.display;
rho_1 = opts.rho_1;
rho_0 = opts.rho_0;
batch = 1;
gamma1 = 5/n;
gamma2 = 1/n;
gamma3 = 5/n;
p_0 = 100;
B = 100;
rr_alpha = zeros(Niter,1);
rr_mu = zeros(Niter,1);
rr_niu = zeros(Niter,1);
diff = zeros(Niter,1);
%iter = 1;
%VI parameter needed
a = 0.5*log(rho_0/rho_1) + 2*log(p);
xty = X'*y;
for kk = 1:Niter
   
    %Update mu, niu
    G_mu = zeros(p,1);
    G_rho = zeros(p,1);
    G_alpha = zeros(p,1);
    
    
    niu = log(1+exp(rho));
    zeta = 1./(1+ exp(-alpha));
    alpha_old = alpha;
    niu_old = niu;
    mu_old = mu;
    eps3 = gamma3; 
    coord_set = datasample(1:1:p, p_0, 'Replace', false);
    mini_batch  = datasample(1:1:n, B);
    z = randn(p,1);
    theta = mu + niu.* z;
    for jj = 1:p_0
        for i = 1:batch
            delta = rand(p,1) < zeta;
            delta1 = delta;delta1(coord_set(jj)) = 1;
            delta0 = delta;delta0(coord_set(jj)) = 0;
            temp1 = log(1+exp(X(mini_batch,delta1)*(theta(delta1))));
            temp0 = log(1+exp(X(mini_batch,delta0)*(theta(delta0))));
        %G_alpha = (0.5*(rho_0-rho_1)*(mu.^2 + niu.^2) + (log(niu)+0.5*log(2*pi) + 0.5) + mu.*(X'*y)...
             %- (X'*(temp./(1+temp)).*theta)).* partial_alpha;
        %G_alpha = (0.5*(rho_0-rho_1)*(mu.^2 + niu.^2)+ mu.*(X'*y)...
             %- (X'*(temp./(1+temp)).*theta)).* partial_alpha;
            G_alpha(coord_set(jj)) = (0.5*(rho_0-rho_1)*(mu(coord_set(jj))^2 + niu(coord_set(jj))^2) -a -alpha(coord_set(jj))... 
            + mu(coord_set(jj))*xty(coord_set(jj))+ n/B * sum(temp0-temp1))* exp(alpha(coord_set(jj)))/((1 + exp(alpha(coord_set(jj)))^2)); 
        end
            G_alpha(coord_set(jj)) = G_alpha(coord_set(jj))/batch;
            alpha(coord_set(jj)) = alpha(coord_set(jj)) + eps3 * G_alpha(coord_set(jj));
            %partial_alpha(j) = exp(alpha(j))/(1+ exp(alpha(j)));
            zeta(coord_set(jj)) = 1/(1+exp(-alpha(coord_set(jj))));
    end
    
    
    %update mu
    for i = 1: batch
        z = randn(p,1);
        theta = mu + niu.* z;
        delta = rand(p,1) < zeta;
        temp = exp(X(mini_batch,delta)*(theta(delta)));
        %G_mu = dot(delta,X'*(y-temp./(1+temp))) - rho_1*(delta.*mu) - rho_0*((1-delta).*mu);
        G_mu(delta) = (n/B)*X(mini_batch,delta)'*(y(mini_batch)-temp./(1+temp))- (rho_1 - rho_0)*(zeta(delta).*mu(delta)) - rho_0*mu(delta);
    end
    G_mu = G_mu/batch;
    eps1 = gamma1;%/(1+kk^0.55);
    mu = mu + eps1 * G_mu; 
    
    %update niu
    logit_rho = exp(rho)./(1+exp(rho));
    for i = 1: batch
        z = randn(p,1);
        theta = mu + niu.* z;
        %delta = rand(p,1) < zeta;
        temp = exp(X(mini_batch,delta)*theta(delta));
        %G_rho = ((-X'*(temp./(1+temp)).*(z.*delta)) -rho_1*(delta.*niu) - rho_0*((1-delta).*niu) + delta./niu).*logit_rho;
        G_rho(delta) = ((-n/B * X(mini_batch,delta)'*(temp./(1+temp))).*z(delta) -(rho_1-rho_0)*(zeta(delta).*niu(delta)) ...
            - rho_0*niu(delta) + 1./niu(delta)).*logit_rho(delta);
    end
    G_rho = G_rho/batch;  
    eps2 = gamma2;%/(1+kk^0.55);
    rho = rho + eps2 * G_rho;
    niu = log(1+exp(rho));
    
    rr_mu(kk) = norm(mu - mu_old)/norm(mu_old);
    rr_niu(kk) = norm(niu - niu_old)/norm(niu_old);
    rr_alpha(kk) = norm(alpha - alpha_old)/norm(alpha_old);
    
    
    for j = 1: batch
        theta = mu + niu .* randn(p,1);
        %for i = 1: n
        %    temp =  temp + y(i)*dot(X(i,:),theta.*delta) - log(1+exp(dot(X(i,:),theta.*delta)));
        %end
        diff(kk) = diff(kk) + norm(theta.*delta-beta_true,2);
    end
    diff(kk) = diff(kk)/(batch*norm(beta_true,2));
    if max([rr_mu(kk),rr_niu(kk),rr_alpha(kk)]) < 0.0025
        stop = kk;
        break
    end
    if display ~= 0
        diff(kk)
    end
    %iter = kk;
end
end