clear all
close all
clc

load net_trained
load aus_data

S=SuspectedPopulation;
E=Exposed;
Q=Quarantine;
I=Infected;
H=Hospitalized;
D=Deaths;
R=Recovered;

y=[S,E,Q,I,H,D,R];

dt=T/length(S); 
T=10; 
t=(0:dt:T)';
Time=t(1:end-1);

input_ref=zeros(length(S),14);
out_pred=zeros(length(S)-1,7);

for ind=1:length(S)-1
    input_ref(ind,1:7)=y(ind,:);
    input_ref(ind,8:14)=y(ind+1,:);
end


for jj=1:length(S)
    y0 = net_trained(input_ref(jj,:)');
    out_pred(jj,:) = y0'; 
end

%% Plotting Results
figure
subplot(2,1,1)
plot(Time,input_ref(:,1),Time,input_ref(:,2),Time,input_ref(:,3),Time,input_ref(:,4),Time,input_ref(:,5),Time,input_ref(:,6),Time,input_ref(:,7));
xlabel('time(sec)');
ylabel('Number');
legend('Susceptible','Exposed','Quarantene','Infected','Hospitalized','Detected','Recovered');
title('Reference Population Data')

subplot(2,1,2)
plot(Time,out_pred(:,1),Time,out_pred(:,2),Time,out_pred(:,3),Time,out_pred(:,4),Time,out_pred(:,5),Time,out_pred(:,6),Time,out_pred(:,7));
xlabel('time(sec)');
ylabel('Rate');
legend('P','\beta_s','\omega_q','\alpha_e','\delta_i','\gamma','\epsilon_h');
title('Predicted Rates')








