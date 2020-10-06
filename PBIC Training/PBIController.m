function [T] = PBIController(z,p,th_d,w_d)
%Controller that uses DB-IC to calculate the torque needed
% th_d = theta_desired
% w_d = omega_desired

th1 = z(1);
th2 = z(3);
thdot1 = z(2);
thdot2 = z(4);

%Current disturbance force on end effector
FxCurrent = p.Fx;
FyCurrent = p.Fy;

%Current Target
xCurrentTar = p.xtarget;
yCurrentTar = p.ytarget;

xdotCurrentTar = 0;
ydotCurrentTar = 0;


Kd = 10;
Kp = 6;
%Torque to track our desired point
T = [Kp*(th_d(1)-th1)+Kd*(w_d(1)-thdot1), Kp*(th_d(1)-th2)+Kd*(w_d(2)-thdot2)];


%Add gravity compensation
T(1) = T(1) + GravityCompT1(0,0,p.d1,p.d2,p.g,p.l1,p.l2,p.m1,p.m2,th1,th2,thdot1,thdot2);
T(2) = T(2) + GravityCompT2(0,0,p.d2,p.g,p.l1,p.l2,p.m2,th1,th2,thdot1);


end

