clc;clear;
load('End_Effector_data');

Traj = [traj_x(1,:)',traj_y(1,:)'];
End_Efftor = [EndEff_x(1,:)',EndEff_y(1,:)'];

diff_x = EndEff_x - traj_x;
diff_y = EndEff_y - traj_y;
RMS_x = rms(diff_x);
RMS_y = rms(diff_y);

diff_q1 = q1_real - q1_ideal;
diff_q2 = q2_real - q2_ideal;
RMS_q1 = rms(diff_q1);
RMS_q2 = rms(diff_q2);

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

figure(4);
subplot(2,1,1);
plot(q1_real);
subplot(2,1,2);
plot(q1_ideal);

figure(5);
subplot(2,1,1);
plot(q2_real);
subplot(2,1,2);
plot(q2_ideal);




