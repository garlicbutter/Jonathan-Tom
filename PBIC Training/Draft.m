clc;clear;
load('End_Effector_data');
%% 
Traj = [traj_x(1,:)',traj_y(1,:)'];
End_Efftor = [EndEff_x(1,:)',EndEff_y(1,:)'];
subplot(2,1,1);
plot(EndEff_x,traj_x);
subplot(2,1,2);
plot(EndEff_y,traj_y);
figure(2);
plot(Traj,End_Efftor);
legend('Traj','End');
hold on;

%% 
R = 1.8;
p.traj = [R*cos(pi/2),R*sin(pi/2);R*cos(pi/2+0.8*pi),R*sin(pi/2+0.8*pi);R*cos(pi/2+1.6*pi),R*sin(pi/2+1.6*pi);R*cos(pi/2+0.4*pi),R*sin(pi/2+0.4*pi);R*cos(pi/2+1.2*pi),R*sin(pi/2+1.2*pi)];

traj = Trajectory_planner(p);
iter = 0;
iterlen = length(traj);

plot(traj(:,1),traj(:,2));

%% 
iterlen = size(Traj);
iterlen = iterlen(1,1);

