clear;
clc;

ori_t = linspace(0,5,25);
x0(1,:)=1*cos(ori_t);
x0(2,:)=1.5*sin(ori_t);
v0(1,:)=-sin(ori_t);
v0(2,:)=1.5*cos(ori_t);
K=1;
B=1;
M=1;
F_int_x=2+ori_t;
F_int_y=2+ori_t;

t0=0;	            % initial time
tfinal=5000;	        % final time
q0=[0 0 0 0]';	    % column vector of initial conditions
tspan=[t0 tfinal]';	% tspan can contain other specific points of integration.
option=odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tau,x] = ode45(@(tau,x)myode(tau,x,x0,v0,K,B,M,F_int_x,F_int_y,ori_t), tspan, q0, option);

plot(x(1,:),x(3,:));
hold on;

% plot(x0(1,:),x0(2,:));
% legend('solved','plan');

%  function dydt = myode(t,y,ft,f,gt,g)
% f = interp1(ft,f,t); % Interpolate the data set (ft,f) at time t
% g = interp1(gt,g,t); % Interpolate the data set (gt,g) at time t
% dydt = -f.*y + g; % Evaluate ODE at time t
%  end
 
F_int_x = interp1(ori_t,F_int_x,tau);


 function dxdt = myode (tau,x,x0,v0,K,B,M,F_int_x,F_int_y,ori_t)
 
F_int_x1 = interp1(ori_t,F_int_x,tau);
save('F_int_x.mat','F_int_x1','F_int_x');
F_int_y = interp1(ori_t,F_int_y,tau);
x_0 = interp1(ori_t,x0(1,:),tau);
y_0 = interp1(ori_t,x0(2,:),tau);
v0_x = interp1(ori_t,v0(1,:),tau);
v0_y = interp1(ori_t,v0(2,:),tau);

dxdt = zeros(4,1);
dxdt(1) = x(2);
dxdt(2) = (K*(x_0 - x(1)) + B*(v0_x-x(2)) - F_int_x1)/M;
dxdt(3) = x(4);
dxdt(4) = (K*(y_0 - x(3)) + B*(v0_y-x(4)) - F_int_y)/M;

 end
 
