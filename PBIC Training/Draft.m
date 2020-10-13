clc;clear;
load('End_Effector_data');

Traj = [traj_x(1,:)',traj_y(1,:)'];
End_Efftor = [EndEff_x(1,:)',EndEff_y(1,:)'];
% iterlen = size(Traj);
% iterlen = iterlen(1,1);
diff_x = EndEff_x - traj_x;
diff_y = EndEff_y - traj_y;
RMS_x = rms(diff_x);
RMS_y = rms(diff_y);

figure(1);
subplot(2,1,1);
plot(EndEff_x);
subplot(2,1,2);
plot(traj_x);

figure(2);
subplot(2,1,1);
plot(EndEff_y);
subplot(2,1,2);
plot(traj_y);

figure(3);
plot(End_Efftor(:,1),End_Efftor(:,2),'r');
legend('Ideal Trajecotry');
hold on;
plot(Traj(:,1),Traj(:,2),'b');
legend('End Effedctor Trajectory');





