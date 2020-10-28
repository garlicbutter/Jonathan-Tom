function [T] = PBIController(z,p,traj,iter,dt_phy)
%Controller that uses PB-IC to calculate the torque needed
%traj = desired trajectory (x,y,z)
%iter = iteration number in the trajectory 

if iter==1 || iter==2 %temporary solution to make every index excutable
    iter = 3;
end

th1 = z(1);
th2 = z(3);
thdot1 = z(2);
thdot2 = z(4);

J = Velocity_transformation(p.l1,p.l2,th1,th2); % J  =Jacobian
J_dt = Jdt(p.l1,p.l2,th1,th2,thdot1,thdot2);
H=[p.I1 0;0 p.I2;];

x_0 = traj(iter,1); %desired position
y_0 = traj(iter,2);
xdot_0 = (traj(iter,1)-traj(iter-1,1))/dt_phy; %desired velocity
ydot_0 = (traj(iter,2)-traj(iter-1,2))/dt_phy;
xa_d = (traj(iter,1)-2*traj(iter-1,1)+traj(iter-2,1))/(dt_phy^2); %desired acceleration
ya_d = (traj(iter,2)-2*traj(iter-1,2)+traj(iter-2,2))/(dt_phy^2);

% q_d = InverseKin(p.l1, p.l2, p.xtarget, p.ytarget); % joint value desirexd







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K = p.K;
B = p.B;
M = p.M;
load('Real_time_F_int_data.mat','F_int_x','F_int_y','iter','dt_phy');
ori_t = 0:dt_phy:iter; 
% F_int_x = F_int_x;
% F_int_y = F_int_y;

t0=0;	            % initial time
tfinal=length(ori_t);	        % final time
q0=[x_0(1) y_0(1) xdot_0(1) ydot_0(1)]';	    % column vector of initial conditions
tspan=[t0 tfinal]';	% tspan can contain other specific points of integration.
option=odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
 [tau,x] = ode45(@(tau,x)myode(tau,x,[x_0, y_0],[xdot_0, ydot_0],,K,B,M,F_int_x,F_int_y,ori_t), tspan, q0, option);
 
q_d = InverseKin(p.l1, p.l2, x(1,end), x(3,end));
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
 
 
 
 
 


[q1_d, q2_d] = InverseKin(p.l1, p.l2, p.xtarget, p.ytarget); % joint value desired
solution_select = 1;
q_d = [q1_d(solution_select)-pi/2, q2_d(solution_select)];

qdt_d = pinv(J)*[xdot_0 ydot_0 0]';% joint value derivative desired
qddt_d =pinv(J)*[xa_d ya_d 0]'+J_dt'*[xdot_0 ydot_0 0]';% joint value second derivative desired


%Torque to track our desired point
T = H*(qddt_d+p.Kd*(qdt_d'-[thdot1 thdot2]')+p.Kp*(q_d'-[th1 th2]'));
T = T+ -J'*[p.Fx; p.Fy;0];

%Add gravity compensation
T(1) = T(1) + GravityCompT1(0,0,p.d1,p.d2,p.g,p.l1,p.l2,p.m1,p.m2,th1,th2,thdot1,thdot2);
T(2) = T(2) + GravityCompT2(0,0,p.d2,p.g,p.l1,p.l2,p.m2,th1,th2,thdot1);


end

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
dxdt(3) = x(3);
dxdt(4) = (K*(y_0 - x(3)) + B*(v0_y-x(4)) - F_int_y)/M;

 end