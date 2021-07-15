function lam=Jacobianizer(y,k)
j=1

P=j*1
betas=j*1
omegaq=j*1
alphae=j*1
deltai=j*1
gamma=j*1
epsilonh=j*1


u=[-1*betas,0,2*y(k,3),0,0,0,0;
    betas,-2*y(k,2),0,0,0,0,0;
    0,2*y(k,2)-alphae,-2*y(k,3),0,0,0,0;
    0,alphae,omegaq,-2*y(k,4),0,0,0;
    0,0,0,deltai,-2*y(k,5),0,0;
    0,0,0,2*gamma*y(k,4)-deltai,epsilonh,0,0;
    0,0,0,gamma*y(k,5)+2*y(k,4)-deltai-2*gamma*y(k,4),gamma*(y(k,4)-deltai),0,0];
lam=max(real(eig(u)));
end