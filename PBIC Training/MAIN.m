% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Two link robot arm with several control method 
% 
% Files:
% MAIN - Execute this file; parameters here.
% 
% Plotter -- Handles all integration and graphics (since the two are linked
% in the live-integration version)
% 
% deriverRelativeAngles -- does symbolic algebra to derive the dynamics and
% the control equations. Automatically writes these to MATLAB functions:
%  - ForwardKin, GravityCompT1, GravityCompT2, Thdotdot1, Thdotdot2
%  All these are autogenerated and should not be
%  directly altered.
% 
% Controllers and plotters are in separate file according to their control
% type (PBIC/DBIC/PID). For each plotter, you can choose toggle the wall 
% and the inverse kinematics solution display.
%
% Trajectory_planner - it returns a trajectory array (x,y) with a minimum jerk.
%
% Difference_Analysis - You can plot several information of the whole
% motion after running the plotter. You have to wait until the robot finish
% the full round of the trajectory.
%
% TODO: 1. Inverse kinematics solution selector.
%       2. 
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
p.Kp = 15; % for PID,PBIC
p.Kd = 5; % for PID,PBIC
p.K = 25*2; % for DBIC,PBIC, K stiffness coeff
p.B = 10*1.3; % for DBIC,PBIC, B damping coeff
p.M = 0.1; % for DBIC, M inertia coeff
    
controller_type = "PID"; %DBIC/ PBIC/ PID

%%%%%%%% trajectory %%%%%%%%%%
% See trajectory_example for more information
% trapezoid 
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
[th1_init, th2_init] = InverseKin(p.l1,p.l2,p.traj(1,1),p.traj(1,2));
p.invKsol = 1; % inverse Kinematics solution number
p.init = [th1_init(p.invKsol)-pi/2    0.0    th2_init(p.invKsol)  0.0]'; %Initial conditions:
endZ = ForwardKin(p.l1,p.l2,p.init(1),p.init(3));
x0 = endZ(1); %End effector initial position in world frame.
y0 = endZ(2);
p.xtarget = x0; %What points are we shooting for in WORLD SPACE?
p.ytarget = y0;

%%%%%%%% wall parameters %%%%%%%%%
p.wall = false;
p.wallleft = 0.95;
p.wallright = 3;
p.wallstiffness = 300;

%%%%%%%% Run Derivers %%%%%%%%
rederive = false;
if rederive
%If the gain matrix hasn't been found yet, then we assume derivation hasn't
%happened yet.
    deriverRelativeAngles;
    disp('Equations of motion and control parameters derived using relative angles.');
end


%%%%%%%% Integrate %%%%%%%
% Integration is done in real time using symplectic euler like    
if controller_type=="DBIC"     
    DBIC_Plotter(p)
elseif controller_type=="PBIC"  
    PBIC_Plotter(p)
elseif controller_type=="PID"
    PID_Plotter(p)
end