 x0=1;
 v0=2;
 K=1;
 B=2;
 M=1;
 F_int(1)=2;
 F_int(2)=2;
global x0 v0 K B M F_int;

t0=0;	            % initial time
 tfinal=50;	        % final time
 q0=[0 0 0 0]';	    % column vector of initial conditions
 tspan=[t0 tfinal]';	% tspan can contain other specific points of integration.
 option=odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
 [tau,q]=ode45(@ODE455, tspan, q0, option);