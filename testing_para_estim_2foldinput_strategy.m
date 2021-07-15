clear all
close all
clc

load net_trained

OF=zeros(1,100);
para=zeros(length(OF),7);
in=0.1;
InitConds = [in;in;in;in;in;in;in];  

for mm=1:length(OF)
    a=0;
    b=1;
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

P=vect_par(1);
betas=vect_par(2);
omegaq=vect_par(3);
alphae=vect_par(4);
deltai=vect_par(5);
gamma=vect_par(6);
epsilonh=vect_par(7);
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

input_ref=zeros(200,14);

for ind=1:200
    input_ref(ind,1:7)=y(ind,:);
    input_ref(ind,8:14)=y(ind+1,:);
end

out_ref=Para(1:end-1,:);

out_pred=zeros(length(out_ref),7);


for jj=1:length(out_ref)
    y0 = net_trained(input_ref(jj,:)');
    out_pred(jj,:) = y0'; 
end

clc


Time=t(1:end-1);

%% Plotting Results
figure
subplot(3,1,1)
plot(Time,out_ref(:,1),Time,out_ref(:,2),Time,out_ref(:,3),Time,out_ref(:,4),Time,out_ref(:,5),Time,out_ref(:,6),Time,out_ref(:,7));
xlabel('time(sec)');
ylabel('Rate');
legend('P','\beta_s','\omega_q','\alpha_e','\delta_i','\gamma','\epsilon_h');
xlim([0 5])
title('Reference')

subplot(3,1,2)
plot(Time,out_pred(:,1),Time,out_pred(:,2),Time,out_pred(:,3),Time,out_pred(:,4),Time,out_pred(:,5),Time,out_pred(:,6),Time,out_pred(:,7));
xlabel('time(sec)');
ylabel('Rate');
legend('P','\beta_s','\omega_q','\alpha_e','\delta_i','\gamma','\epsilon_h');
xlim([0 5])
title('Estimated')

subplot(3,1,3)
plot(Time,abs(out_ref(:,1)-out_pred(:,1)),Time,abs(out_ref(:,2)-out_pred(:,2)),Time,abs(out_ref(:,3)-out_pred(:,3)),Time,abs(out_ref(:,4)-out_pred(:,4)),Time,abs(out_ref(:,5)-out_pred(:,5)),Time,abs(out_ref(:,6)-out_pred(:,6)),Time,abs(out_ref(:,7)-out_pred(:,7)));
xlabel('time(sec)');
ylabel('Rate');
legend('\epsilon_{P}(t)','\epsilon_{\beta_s}(t)','\epsilon_{\omega_q}(t)','\epsilon_{\alpha_e}(t)','\epsilon_{\delta_i}(t)','\epsilon_{\gamma}(t)','\epsilon_{\epsilon_h}(t)');
xlim([0 5])
title('Absolute Error')