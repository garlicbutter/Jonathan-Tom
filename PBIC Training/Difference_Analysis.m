clc;clear;
load('End_Effector_data');

Traj = [traj_x(1,:)',traj_y(1,:)'];
End_Efftor = [EndEff_x(1,:)',EndEff_y(1,:)'];

diff_x = EndEff_x - traj_x;
diff_y = EndEff_y - traj_y;
RMS_x = rms(diff_x);
RMS_y = rms(diff_y);
RMS = rms([diff_x, diff_y]);

diff_q1 = q1_real - q1_ideal;
diff_q2 = q2_real - q2_ideal;
RMS_q1 = rms(diff_q1);
RMS_q2 = rms(diff_q2);

figure(1);
subplot(2,1,1);
hold on
title("Value of x position in Cartasian");
plot(EndEff_x,'r');
plot(traj_x,'b');
legend("End Effector","Required");
xlabel("time lapse");
ylabel("Value of x");
str_rmsx = {'RMS of x: ',RMS_x};
text(0.3,1,str_rmsx)

subplot(2,1,2);
hold on
title("Value of y position in Cartasian");
plot(EndEff_y,'r');
plot(traj_y,'b');
legend("End Effector","Required");
xlabel("time lapse");
ylabel("Value of y");
str_rmsy = {'RMS of y: ',RMS_y};
text(0.3,1,str_rmsy)

figure(2);
title("trajectory comparison");
hold on
plot(End_Efftor(:,1),End_Efftor(:,2),'r');
plot(Traj(:,1),Traj(:,2),'b');
xlabel("x [m]");
ylabel("y [m]");
legend("End Effector","Required");
% str = {'RMS: ',RMS};
% text(1.3,1,str)

figure(3);
subplot(2,1,1);
hold on
title("Joint Value");
plot(q1_real,'r');
plot(q1_ideal,'b');
legend("Impedance Result","Required");
xlabel("time lapse");
ylabel("Joint value of q1 [rad]");
str_rmsq1 = {'RMS of q1: ',RMS_q1};
text(0,0,str_rmsq1)

subplot(2,1,2);
hold on
plot(q2_real,'r');
plot(q2_ideal,'b');
legend("Impedance Result","Required");
xlabel("time lapse");
ylabel("Joint value of q2 [rad]");
str_rmsq2 = {'RMS of q2: ',RMS_q2};
text(-0.3,1,str_rmsq2)

