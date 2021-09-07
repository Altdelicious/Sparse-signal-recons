#This code file requires the Rice Wavelet Toolbox to be installed in MATLAB.

close all
clc; clear;

f = double(imread('Camera.tif'));
[m, n] = size(f);

%scrsz = get(0,'ScreenSize');
%figure(1)
%set(1,'Position',[10 scrsz(4)*0.05 scrsz(3)/4 0.85*scrsz(4)])

disp('Creating observation operator...');

middle = n/2 + 1;

% Experiment 1 (see paper)
% sigma = 0.56;
% h = zeros(size(f));
% for i=-4:4
%    for j=-4:4
%       h(i+middle,j+middle)= 1; 
%    end
% end


% Experiment 2 (see paper)
sigma = sqrt(2);
h = zeros(size(f));
for i=-4:4
   for j=-4:4
      h(i+middle,j+middle)= (1/(1+i*i+j*j));
   end
end

% Experiment 3 (see paper)
% sigma = sqrt(8);
% h = zeros(size(f));
% for i=-4:4
%    for j=-4:4
%       h(i+middle,j+middle)= (1/(1+i*i+j*j));
%    end
% end

h = fftshift(h);   
h = h/sum(h(:));

R = @(x) real(ifft2(fft2(h).*fft2(x)));
RT = @(x) real(ifft2(conj(fft2(h)).*fft2(x)));

wav = daubcqf(2);
W = @(x) midwt(x,wav,3);
WT = @(x) mdwt(x,wav,3);

%function handles that compute A = RW  and A' =W'*R' 
A = @(x) R(W(x));
AT = @(x) WT(RT(x));

% generate noisy blurred observations
y = R(f) + sigma*randn(size(f));
imwrite(uint8(y), 'blur_image.png')

% regularization parameter
tau = .35;

% set tolA
tolA = 1.e-3;

% Run IST
disp('Starting IST algorithm')
%[theta_ist,~,obj_IST,times_IST,~,mses_IST]= IST(y,A,tau,'AT',AT,'ToleranceA',tolA,'StopCriterion',1,'Initialization',AT(y));
[theta_ist,~,obj_IST,times_IST,~,mses_IST]= IST(y,A,tau,'AT',AT,'ToleranceA',tolA,'StopCriterion',1,'Initialization',AT(y));
temp = W(theta_ist);
imwrite(uint8(temp), 'deblur_image_ist.png')
rmse_ist = norm(temp-y)/norm(y);

disp('Starting GPSR algorithm')
% run the GPSR functions, until they reach the same value
% of objective function reached by IST.
[theta,theta_debias,obj_GPSR_Basic,times_GPSR_Basic,debias_s,mses_GPSR_Basic]= GPSR_Basic(y,A,tau,'AT',AT,'ToleranceA',obj_IST(end),'StopCriterion',4,'Initialization',AT(y),'Verbose',0);
temp = W(theta);
imwrite(uint8(temp), 'deblur_image_theta.png')
rmse_gpsr = norm(temp-y)/norm(y);
disp('GPSR ends')


% ================= Plotting results ==========
figure
subplot(3,1,1)
imagesc(f)
colormap(gray(255))
axis off
axis equal
title('Original image')
subplot(3,1,2)
imagesc(y)
colormap(gray(255))
axis off
axis equal
title('Blurred image')
subplot(3,1,3)
imagesc(W(theta_ist))
colormap(gray)
axis off
axis equal
title('Deblurred image')
saveas(gcf, 'Images.png')


figure
%hold on
%plot(obj_QP_BB_notmono,'r--','LineWidth',1.8);
plot(obj_GPSR_Basic,'b-*')
hold on
plot(obj_IST,'k-')
hold off
legend('GPSR-Basic','IST')
title('Objective function vs Iterations')
ylabel('Objective function')
xlabel('Iterations')
saveas(gcf, 'fig2.png')

figure
%hold on
plot(times_GPSR_Basic,obj_GPSR_Basic,'b-*')
hold on
plot(times_IST,obj_IST,'k-')
hold off
legend('GPSR-Basic','IST')
title('Objective function vs CPU time')
ylabel('Objective function')
xlabel('CPU time (seconds)')
saveas(gcf, 'fig3.png')


%figure(4)
%hold on
%plot(times_GPSR_Basic,mses_GPSR_Basic,'g:')
%plot(times_IST,mses_IST,'m-.')
%hold off
%legend('GPSR-Basic','IST')
%ylabel('Deconvolution MSE','FontName','Times','FontSize',16)
%xlabel('CPU time (seconds)','FontName','Times','FontSize',16)

