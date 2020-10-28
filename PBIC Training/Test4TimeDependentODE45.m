clc
clear
close all

load('End_Effector_data.mat');

Traj = [traj_x(1,:)',traj_y(1,:)'];
End_Efftor = [EndEff_x(1,:)',EndEff_y(1,:)'];
K = 5;
B = 3;
M = 2;

x_0 = EndEff_x(1,50:100)';
y_0 = EndEff_y(1,50:100)';
t=linspace(0,10,length(x_0))';
time_interval = t(2);
xdot_0 = diff(x_0)/time_interval;
ydot_0 = diff(y_0)/time_interval;
xdot_0 = [xdot_0(1); xdot_0];
ydot_0 = [ydot_0(1); ydot_0];

F_int= [t*0.1, t*0.05];

%%
t0=0;             % initial time
tfinal=length(t);         % final time
q0=[x_0(1) y_0(1) xdot_0(1) ydot_0(1)]';     % column vector of initial conditions
tspan=[t0 tfinal]'; % tspan can contain other specific points of integration.
option=odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[tau,q] = ode45(@(tau,x)myode(tau,x,[x_0, y_0],[xdot_0, ydot_0],K,B,M,F_int,t), tspan, q0, option);
%%
close all
figure(1)
hold on
grid on
plot(q(:,1))
plot(q(:,2))
plot(q(:,3))
plot(q(:,4))
plot(x_0,"*")
plot(y_0,"*")
plot(xdot_0,"*")
plot(ydot_0,"*")
plot(F_int(:,1),'.')
plot(F_int(:,2),'.')
legend("x_m","y_m","vx_m","vy_m","x_0","y_0","vx_0","vy_0","Fx","Fy")
%%
function dxdt = myode (tau,x,x0,v0,K,B,M,F_int,t)
 
F_int_x = interp1(t,F_int(:,1),tau);
F_int_y = interp1(t,F_int(:,2),tau);
x_0 = interp1(t,x0(:,1),tau);
y_0 = interp1(t,x0(:,2),tau);
v0_x = interp1(t,v0(:,1),tau);
v0_y = interp1(t,v0(:,2),tau);

dxdt = zeros(4,1);
dxdt(1) = x(2);
dxdt(2) = (K*(x_0 - x(1)) + B*(v0_x-x(2)) - F_int_x)/M;
dxdt(3) = x(4);
dxdt(4) = (K*(y_0 - x(3)) + B*(v0_y-x(4)) - F_int_y)/M;

end