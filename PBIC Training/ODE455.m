function ODE45 = ODE45 (tau,q)
global x0 v0 K B M F_int;
ODE45 = zeros(4,1);
ODE45(1) = q(2);
ODE45(2) = (K*(x0 - q(1)) + B*(v0-q(2)) - F_int(1))/M;
ODE45(3) = q(3);
ODE45(4) = (K*(x0 - q(3)) + B*(v0-q(4)) - F_int(2))/M;