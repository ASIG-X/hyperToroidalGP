clear all; close all;

% simulate data
n_data = 500;
theta = [linspace(0,2*pi,n_data),2*pi];

thetab = pi/3;
conc = [-2;0];

R = [cos(thetab),-sin(thetab);
     sin(thetab),cos(thetab)];
bing = BinghamDistribution(conc,R);

ax = [0,pi/3,-2*pi/3];
kappa = [3,10,10];
vmf = cell(1,size(ax,2));
for i = 1:size(ax,2)
  vmf{i} = VMDistribution(ax(i),kappa(i));
end
w = ones(1,length(vmf))/length(vmf);
mixPdfvM = CircularMixture(vmf,w);

fh_f = @(x) 2*mixPdfvM.pdf(x) + 2*bing.pdf([cos(x);sin(x)]);
f = fh_f(theta);

n_smp = 20;
gauss = GaussianDistribution(0,0.05^2);
idx_smp = [10,20,50,55,60,70,75,100,110,140,200,210,250,310,320,355,370,388,400,450];
training_points = theta(idx_smp);
obs = f(idx_smp) + gauss.sample(size(idx_smp,2));
idx_te = setdiff(1:(n_data+1),idx_smp);
test_points = theta(idx_te);
gt = f(idx_te);

x0 = [1,1,1,0.1];
[xeopt_vM] = optimizationMOGP(1,1,1,x0,{'vMKernel'},training_points,obs);
[m_est_vM,covar_vM] = predictionMOGP(1,1,1,xeopt_vM,{'vMKernel'},training_points,obs,test_points);
covar_vM = covar_vM - eye(size(covar_vM,1))*xeopt_vM.noiseParam^2;
[x_est,y_est] = pol2cart(test_points,m_est_vM'+1);
lb = m_est_vM - 2*sqrt(diag(covar_vM));
ub = m_est_vM + 2*sqrt(diag(covar_vM));

co = 0.8;
fig1 = figure;
[x_lb,y_lb] = pol2cart(test_points,lb'+1);
[x_ub,y_ub] = pol2cart(test_points,ub'+1);
inBetween = [y_lb,fliplr(y_ub)];
t2 = [x_lb,fliplr(x_ub)];
fill(t2,inBetween, 'k','FaceAlpha',0.2,'EdgeColor','none');  
hold on
p1 = plot(co*cos(theta),co*sin(theta),'--k','LineWidth',2);
p1.Color(4) = 1;
[x_gt,y_gt] = pol2cart(theta,f+1);
plot(x_gt,y_gt,'Color','#FF0000','LineWidth',2);
axis equal
[x_tr,y_tr] = pol2cart(training_points,obs+1);
pe1 = plot(x_est,y_est,'b','LineWidth',2);
pe1.Color(4) = 0.7;
scatter(x_tr,y_tr,50,'filled','MarkerFaceColor','#3f9b0b')
axis off

[xeopt_SE] = optimizationMOGP(1,1,1,x0,{'SEKernel'},training_points,obs);
[m_est_SE,covar_SE] = predictionMOGP(1,1,1,xeopt_SE,{'SEKernel'},training_points,obs,test_points);
covar_SE = covar_SE - eye(size(covar_SE,1))*xeopt_SE.noiseParam^2;
[x_est,y_est] = pol2cart(test_points,m_est_SE'+1);
lb = m_est_SE - 2*sqrt(diag(covar_SE));
ub = m_est_SE + 2*sqrt(diag(covar_SE));

fig2 = figure;
[x_lb,y_lb] = pol2cart(test_points,lb'+1);
[x_ub,y_ub] = pol2cart(test_points,ub'+1);
inBetween = [y_lb,fliplr(y_ub)];
t2 = [x_lb,fliplr(x_ub)];
fill(t2,inBetween, 'k','FaceAlpha',0.2,'EdgeColor','none');  
hold on
p2 = plot(co*cos(theta),co*sin(theta),'--k','LineWidth',2);
p2.Color(4) = 1;

[x_gt,y_gt] = pol2cart(theta,f+1);
plot(x_gt,y_gt,'Color','#FF0000','LineWidth',2);

axis equal
pe2 = plot(x_est,y_est,'b','LineWidth',2);
pe2.Color(4) = 0.7;
scatter(x_tr,y_tr,50,'filled','MarkerFaceColor','#3f9b0b')
axis off