clear all
close all 
clc
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
 J=zeros(length(t),1);
 
 for n=1:length(J)
     J(n)=Jacobianizer(y,n);
 end


plot(t,y(:,1),t,y(:,2),t,y(:,3),t,y(:,4),t,y(:,5),t,y(:,6),t,y(:,7),t,sum(InitConds)*ones(length(t),1),'*');
xlabel('days')
ylabel('Number')
grid on
legend('Susceptible','Exposed','Quarantene','Infected','Hospitalized','Detected','Recovered', 'Total Number')

figure 
plot(t,J,t,zeros(length(J),1),'*');
xlabel('days')
ylabel('Maximal Eigenvalue')
ylim([-4 4])
grid on
    
 x1=y(:,4);
 x2=y(:,5);
fun = @(C)sum(abs(C(1)-x2.^2+(C(2)-C(3)*C(4))*x2+C(3)*x1.*x2).^2);
C0 = [0.1,0.1,0.1,0.1];
C = fminsearch(fun,C0);
est_P=C(1);
est_epsilonh=C(2);
est_gamma=C(3);
est_deltai=C(4);

clc
% Method 1: Define each column
T = table([P;est_P],[epsilonh;est_epsilonh],[gamma;est_gamma],[deltai;est_deltai],'VariableNames',{'P','epsilonH','gamma','deltaI'},'RowName',{'Ref','Est'}); 
disp(T) 