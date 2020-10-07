function [T] = PBIController(z,p,traj,iter)
%Controller that uses DB-IC to calculate the torque needed
%traj = desired trajectory (x,y,z)
%iter = iteration number in the trajectory 

th1 = z(1);
th2 = z(3);
thdot1 = z(2);
thdot2 = z(4);
x_d = traj(iter,1); %desired position
y_d = traj(iter,2);
xv_d = (traj(iter,1)-traj(iter-1,1))/dt; %desired velocity
yv_d = (traj(iter,1)-traj(iter-1,1))/dt;
q_d = InverseKin(p.l1, p.l2, p.xtarget, p.ytarget) % joint value desired
qdt_d = ;% joint value derivative desired
qddt_d = ;% joint value second derivative desired


% J  =Jacobian
J = Velocity_transformation(p.l1,p.l2,th1,th2);

%Torque to track our desired point
T = H*[qddt_d+p.Kd*([qdt_d 0]'-[thdot1 thdot2 0]')+p.Kp*([q_d 0]'-[th1 th2 0]')];
T = T+ -J'*[p.Fx; p.Fy;0];

%Add gravity compensation
T(1) = T(1) + GravityCompT1(0,0,p.d1,p.d2,p.g,p.l1,p.l2,p.m1,p.m2,th1,th2,thdot1,thdot2);
T(2) = T(2) + GravityCompT2(0,0,p.d2,p.g,p.l1,p.l2,p.m2,th1,th2,thdot1);


end

