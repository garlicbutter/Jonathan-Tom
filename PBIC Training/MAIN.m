% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two link robot arm with control to track a point that the user clicks.
% 
% Files:
% MAIN - Execute this file; parameters here.
% 
% Plotter -- Handles all integration and graphics (since the two are linked
% in the live-integration version)
% 
% FullDyn -- Dynamics function of the form zdot = dynamics(z,t,params).
% This function evaluates BOTH the controller and the dynamics. Can
% probably be crunched through ode45 if you don't care about user
% interaction.
%   
% deriverRelativeAngles -- does symbolic algebra to derive the dynamics and
% the control equations. Automatically writes these to MATLAB functions:
%   - ForwardKin, GravityCompT1, GravityCompT2, ImpedanceControl,
%   Thdotdot1, Thdotdot2 --- All these are autogenerated and should not be
%   directly altered.
% 
% 
% Matthew Sheen, 2014
% 
% also:
% Jonathan Oh, 2020
% Zitao, Yu, 2020
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear;

%%%%%%%% Control Parameters %%%%%%%%
%Controller Gains and type
p.Kp = 200; % for PID,PBIC
p.Kd = 20; % for PID,PBIC
p.K = 45; % for DBIC,PBIC, K stiffness coeff
p.B = 22; % for DBIC,PBIC, B damping coeff
p.M = 0.08; % for DBIC,PBIC, M inertia coeff

controller_type = "DBIC"; %DBIC/ PBIC/ PID

%%%%%%%% trajectory %%%%%%%%%%

%octagon trajectory
% p.traj = [-1.5, 0;-1, 1; 0, 1.5; 1, 1;1.5, 0; 1, -1; 0, -1.5;-1,-1];

%star pentagon trajectory
% R = 1.8;
% p.traj = [R*cos(pi/2),R*sin(pi/2);R*cos(pi/2+0.8*pi),R*sin(pi/2+0.8*pi);R*cos(pi/2+1.6*pi),R*sin(pi/2+1.6*pi);R*cos(pi/2+0.4*pi),R*sin(pi/2+0.4*pi);R*cos(pi/2+1.2*pi),R*sin(pi/2+1.2*pi)];

%trapezoid trajectory
R = 0.5;
p.traj = [1*R,3*R; 3*R,2*R; 3*R,-2*R; 1*R,-3*R]; 


%%%%%%%% System Parameters %%%%%%%%
p.g = 9.81;
p.m1 = 1; %Mass of link 1.
p.m2 = 1; %Mass of link 2.
p.l1 = 1; %Total length of link 1.
p.l2 = 1; %Total length of link 2.
p.d1 = p.l1/2; %Center of mass distance along link 1 from the fixed joint.
p.d2 = p.l2/2; %Center of mass distance along link 2 from the fixed joint.
p.I1 = 1/12*p.m1*p.l1^2; %Moment of inertia of link 1 about COM
p.I2 = 1/12*p.m2*p.l2^2; %Moment of inertia of link 2 about COM
p.Fx = 0;
p.Fy = 0;
[tmp1, tmp2] = InverseKin(p.l1,p.l2,p.traj(1,1),p.traj(1,2));
p.init = [tmp1(1)-pi/2    0.0    tmp2(1)  0.0]'; %Initial conditions:
endZ = ForwardKin(p.l1,p.l2,p.init(1),p.init(3));
x0 = endZ(1); %End effector initial position in world frame.
y0 = endZ(2);
p.xtarget = x0; %What points are we shooting for in WORLD SPACE?
p.ytarget = y0;

%%%%%%%% Run Derivers %%%%%%%%
rederive = false;
if rederive
%If the gain matrix hasn't been found yet, then we assume derivation hasn't
%happened yet.
        deriverRelativeAngles;
        disp('Equations of motion and control parameters derived using relative angles.');
end


%%%%%%%% Integrate %%%%%%%
if controller_type=="DBIC"     
    DBIC_Plotter(p) %Integration is done in real time using symplectic euler like we did in the CS animation class.    
elseif controller_type=="PBIC"  
    PBIC_Plotter(p)
elseif controller_type=="PID"
    PID_Plotter(p)
end