function [T] = DBIController(z,p)
%Controller that uses DB-IC to calculate the torque needed

th1 = z(1);
th2 = z(3);
thdot1 = z(2);
thdot2 = z(4);
q = [th1 th2]'; %joint value
dq = [thdot1 thdot2]'; 
x_o = ;%virtual value
xd_o = ; 

%Current disturbance force on end effector
FxCurrent = p.Fx;
FyCurrent = p.Fy;

%Current Target
xCurrentTar = p.xtarget;
yCurrentTar = p.ytarget;

xdotCurrentTar = 0;
ydotCurrentTar = 0;

% J  =Jacobian
J = Velocity_transformation(p.l1,p.l2,th1,th2);
J_dt = Jdt(p.l1,p.l2,th1,th2,thdot1,thdot2);
H=[p.I1 0;0 p.I2;];
Lq = ForwardKin(p.l1,p.l2,th1,th2);

K = p.Kp;
B = p.Kd;
F_int = K*() + B*() - M*() % K stiffness, B damping, M inertia

%Torque to track our desired point

temp = K*(x_o-Lq)+B*(xd_o-J*dq)-F_int;
T = H*J'*(cross(inv(M),temp)-J_dt*dq);
T = T+ -J'*F_int;

%Add gravity compensation
T(1) = T(1) + GravityCompT1(0,0,p.d1,p.d2,p.g,p.l1,p.l2,p.m1,p.m2,th1,th2,thdot1,thdot2);
T(2) = T(2) + GravityCompT2(0,0,p.d2,p.g,p.l1,p.l2,p.m2,th1,th2,thdot1);


end

