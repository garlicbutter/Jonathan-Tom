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
RMS = rms([diff_x, diff_y]);

%%%%%%
figure(1);
subplot(2,1,1);
hold on
title("Link 1 Joint value");
plot(EndEff_x,'r');
plot(traj_x,'b');
legend("End Effector","Required");
xlabel("time lapse");
ylabel("joint value [rad]");

subplot(2,1,2);
hold on
title("Link 2 Joint value");
plot(EndEff_y,'r');
plot(traj_y,'b');
legend("End Effector","Required");
xlabel("time lapse");
ylabel("joint value [rad]");

%%%%%%
figure(2);
title("trajectory comparison");
hold on
plot(End_Efftor(:,1),End_Efftor(:,2),'r');
plot(Traj(:,1),Traj(:,2),'b');
xlabel("x [m]");
ylabel("y [m]");
legend("End Effector","Required");
str = {'RMS: ',RMS};
text(1.3,1,str)





