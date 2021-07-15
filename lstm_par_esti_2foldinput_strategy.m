clear all
close all 
clc


dt=0.05; 
T=10; 
t=0:dt:T;
dim_in=14;
dim_out=7;
NofT = 100; % Number of trajectories (learning data)
iterations=20;
%%
%load parameter_estim_SITR
%%
% matrices for machine learning
input = zeros(NofT *iterations* T / dt,dim_in) ;
output = zeros(NofT *iterations* T / dt,dim_out) ;

for iter=1:iterations 
    
    
    %% Module#1: Generate Suitable Parameters out of mess
    OF=zeros(1,100);
    para=zeros(length(OF),7);
    in=0.1;
    InitConds = [in;in;in;in;in;in;in]; 

    for mm=1:length(OF)
        a=0;
        b=0.5;
        P= (b-a)*rand + a;
        betas= (b-a)*rand + a;
        omegaq= (b-a)*rand + a;
        alphae= (b-a)*rand + a;
        deltai= (b-a)*rand + a;
        gamma= (b-a)*rand + a;
        epsilonh= (b-a)*rand + a;
        para(mm,:)=[P,betas,omegaq,alphae,deltai,gamma,epsilonh];

        dt=0.05; 
        T=10; 
        t=0:dt:T;

        SEQHIRD = @(t,x) ([ P-betas*x(1)+(x(3)-omegaq)*x(3)
                                    betas*x(1)-x(2)^2
                                    -(x(3))^2+(x(2)-alphae)*x(2)
                                    alphae*x(2)+omegaq*x(3)-x(4)^2
                                    deltai*x(4)-x(5)^2
                                    epsilonh*x(5)+gamma*(x(4)-deltai)*x(4)
                                    gamma*(x(4)-deltai)*x(5)+(x(4)-deltai-gamma*(x(4)-deltai))*x(4)]);



        [t,y] = ode45(SEQHIRD, t, InitConds);
        OF(mm)=sum(abs(y(:,1))+abs(y(:,2))+abs(y(:,3))+abs(y(:,4))+abs(y(:,5))+abs(y(:,6))+abs(y(:,7)));
    end
    [CC,K]=min(OF);
    vect_par=para(K,:);
    
    % Finally some useful parameters
    P=vect_par(1);
    betas=vect_par(2);
    omegaq=vect_par(3);
    alphae=vect_par(4);
    deltai=vect_par(5);
    gamma=vect_par(6);
    epsilonh=vect_par(7);
     
    % Some time specifications
    dt=0.05; 
    T=10; 
    t=0:dt:T;
    
    % Vecotrizing the parameters
    P_v=P*ones(length(t),1);
    betas_v=betas*ones(length(t),1);
    omegaq_v=omegaq*ones(length(t),1);
    alphae_v= alphae*ones(length(t),1);
    deltai_v=deltai*ones(length(t),1);
    gamma_v=gamma*ones(length(t),1);
    epsilonh_v=epsilonh*ones(length(t),1);
    
    
    % Parameter matrix
    Para=[P_v,betas_v,omegaq_v,alphae_v,deltai_v,gamma_v,epsilonh_v];
    
    
    % Defining SEQHIRD model on constant parameters
    SEQHIRD = @(t,x) ([ P-betas*x(1)+(x(3)-omegaq)*x(3)
                            betas*x(1)-x(2)^2
                            -(x(3))^2+(x(2)-alphae)*x(2)
                            alphae*x(2)+omegaq*x(3)-x(4)^2
                            deltai*x(4)-x(5)^2
                            epsilonh*x(5)+gamma*(x(4)-deltai)*x(4)
                            gamma*(x(4)-deltai)*x(5)+(x(4)-deltai-gamma*(x(4)-deltai))*x(4)]);



    for jj=1:NofT 
        in=0.1;
        InitConds = [in+0.01*rand(1);in+0.01*rand(1);in+0.01*rand(1);in+0.01*rand(1);in+0.01*rand(1);in+0.01*rand(1);in+0.01*rand(1)];  
       
        [t,y] = ode45(SEQHIRD, t, InitConds);
        % This is meant when infinities are found in solution
%         if length(y)~=200
%             break
%         end
        for ll=1:7
            input(1 + (jj-1)*1*(T/dt)+(iter-1)*NofT*T/dt:jj*1*T/dt+(iter-1)*NofT*T/dt,ll) = y(1:end-1,ll);
        end
        for ll=8:14
            input(1 + (jj-1)*1*(T/dt)+(iter-1)*NofT*T/dt:jj*1*T/dt+(iter-1)*NofT*T/dt,ll) = y(2:end,ll-7);
        end
        
        output(1 + (jj-1)*1*(T/dt)+(iter-1)*NofT*T/dt:jj*T/dt+(iter-1)*NofT*T/dt,:) = Para(1:end-1,:);

    end


end


%% Defining neural network
numFeatures = 14;
numHiddenUnits = 125;
numResponses = 7;
layers = [sequenceInputLayer(numFeatures) lstmLayer(numHiddenUnits,'OutputMode','sequence') fullyConnectedLayer(numResponses) regressionLayer];

maxEpochs = 60;
miniBatchSize = 20;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'Verbose',0);

net = trainNetwork(input',output',layers,options);

% saving the model
save('net_trained_lstm.mat','net')