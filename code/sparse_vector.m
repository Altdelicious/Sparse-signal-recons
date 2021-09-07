clear; clc; close all;
n = 2^12;
k = 2^10;

n_spikes = [160,320,480,640,800,960];
rmses_ls = zeros(size(n_spikes));
rmses_ist = zeros(size(n_spikes));
rmses_gpsr = zeros(size(n_spikes));

num_nonzero_ls = zeros(size(n_spikes));
num_nonzero_ist = zeros(size(n_spikes));
num_nonzero_gpsr = zeros(size(n_spikes));

i=1;
for i=1:length(n_spikes)
f = zeros(n,1);
q = randperm(n);
f(q(1:n_spikes(i))) = sign(randn(n_spikes(i),1));

R = randn(k,n);
R = orth(R')';

hR = @(x) R*x;
hRt = @(x) R'*x;

sigma = 0.01;
y = hR(f) + sigma*randn(k,1);

tau = 0.1*max(abs(R'*y));

[x_l1_ls,status,history] = l1_ls(R,y,2*tau,0.01);
obj_ls = history(2,1:end)/2;
tolA = history(2,end)/2;

rmses_ls(i) = norm(f-x_l1_ls)/norm(f);
num_nonzero_ls(i) = sum(x_l1_ls~=0);

[x_ist,~,obj_IST,times_IST,~,mses_IST]= IST(y,hR,tau,'AT',hRt,'ToleranceA',tolA,'StopCriterion',4,'Initialization',0);
rmses_ist(i) = norm(f-x_ist)/norm(f);
num_nonzero_ist(i) = sum(x_ist~=0);

%t_l1_ls = history(7,end);

stopCri = 4;
debias = 0;

[x_GPSR_Basic,x_debias_GPSR_Basic,obj_GPSR_Basic,...
    times_GPSR_Basic,debias_start_Basic,mse]= ...
	 GPSR_Basic(y,hR,tau,...
         'Debias',debias,...
         'AT',hRt,... 
         'Initialization',0,...
    	 'StopCriterion',stopCri,...
	     'ToleranceA',tolA,...
         'ToleranceD',0.0001);
t_GPSR_Basic = times_GPSR_Basic(end);
rmses_gpsr(i) = norm(f-x_GPSR_Basic)/norm(f);
num_nonzero_gpsr(i) = sum(x_GPSR_Basic~=0);

end

% ================= Plotting results ==========

figure
hold on
plot(obj_GPSR_Basic,'b-*')
plot(obj_IST,'k')
plot(obj_ls,'g-')
legend('GPSR-Basic', 'ISTA', 'L1-LS')
xlabel('Iterations')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
hold off
saveas(gca, 'fig1.png')


figure
hold on
plot(times_GPSR_Basic,obj_GPSR_Basic,'b-*')
plot(times_IST,obj_IST,'k')
plot(history(7,:),history(2,:)/2,'g-.')
legend('GPSR-Basic', 'ISTA', 'L1-LS')
xlabel('CPU time (seconds)')
ylabel('Objective function')
title(sprintf('n=%d, k=%d, tau=%g',n,k,tau))
saveas(gca, 'fig2.png')
hold off

figure
hold on
plot(n_spikes,rmses_ls,'g-')
plot(n_spikes,rmses_ist,'k')
plot(n_spikes,rmses_gpsr,'b-*')
legend('L1-LS', 'ISTA', 'GPSR-Basic')
xlabel('Number of spikes')
ylabel('RMSE')
title(sprintf('n=%d, k=%d',n,k))
saveas(gca, 'fig3.png')
hold off

figure
hold on
plot(n_spikes,num_nonzero_ls,'g-')
plot(n_spikes,num_nonzero_ist,'k')
plot(n_spikes,num_nonzero_gpsr,'b-*')
legend('L1-LS', 'ISTA', 'GPSR-Basic')
xlabel('Number of spikes')
ylabel('Number of non-zero componenets')
title(sprintf('n=%d, k=%d',n,k))
saveas(gca, 'fig4.png')
hold off
