function [T] = PIDController(z,p,th_d,w_d)
%Controller that uses DB-IC to calculate the torque needed
% th_d = theta_desired
% w_d = omega_desired
th1 = z(1);
th2 = z(3);
thdot1 = z(2);
thdot2 = z(4);

% J  =Jacobian
J = Velocity_transformation(p.l1,p.l2,th1,th2);

%Torque to track our desired point
T = [p.Kp*(th_d(1)-th1)+p.Kd*(w_d(1)-thdot1), p.Kp*(th_d(1)-th2)+p.Kd*(w_d(2)-thdot2)];



%Add gravity compensation
T(1) = T(1) + GravityCompT1(0,0,p.d1,p.d2,p.g,p.l1,p.l2,p.m1,p.m2,th1,th2,thdot1,thdot2);
T(2) = T(2) + GravityCompT2(0,0,p.d2,p.g,p.l1,p.l2,p.m2,th1,th2,thdot1);


end

