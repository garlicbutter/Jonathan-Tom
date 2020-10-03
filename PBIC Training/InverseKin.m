%% system parameters & initial conditions

l1 = 1;
l2 = 1;

x0 = 1.6730;%l1*cos(q1)+l2*cos(q1+q2)
y0 = 0.9659;%l1*sin(q1)+l2*sin(q1+q2)

% q1 = pi/4;
% q2 = -pi/6;

syms q1;
Cq2 = (x0^2+y0^2-l1^2-l2^2)/2/l1;
Sq2 = -sqrt(1-((x0^2+y0^2-l1^2-l2^2)/2/l1)^2);
q2 = atan2(Sq2,Cq2)
syms Sq1 Cq1;
eqns = [x0 == l1*Cq1+l2*(cos(q2)*Cq1-sin(q2)*Sq1), y0 == l1*Sq1 + l2*Sq1*cos(q2)+l2*sin(q2)*Cq1];
q1 = solve(eqns,[Sq1 Cq1]); 
q1 = atan2(double(q1.Sq1),double(q1.Cq1));

%% inverse Kinematics solver
ik = inverseKinematics;

robot = rigidBodyTree('DataFormat','column','MaxNumBodies',3);
L1 = 1;
L2 = 1;
body = rigidBody('link1');
joint = rigidBodyJoint('joint1', 'revolute');
setFixedTransform(joint,trvec2tform([0 0 0]));
joint.JointAxis = [0 0 1];
body.Joint = joint;
addBody(robot, body, 'base');

%% approxiamete plot of the model
f = figure;
set(f,'WindowButtonMotionFcn','','WindowButtonDownFcn',@ClickDown,'WindowButtonUpFcn',@ClickUp,'KeyPressFc',@KeyPress);

% traj = [];
% figData.Fx = [];
% figData.Fy = [];
% figData.xend = [];
% figData.yend = [];
% figData.fig = f;
% figData.tarControl = true;

figData.simArea = subplot(1,1,1); %Eliminated other subplots, but left this for syntax consistency.
axis equal
hold on
grid on

%Create link1:
width1 = l1*0.1;
xdat1 = (0.5*width1)*[-1 1 1 -1]+l1*cos(q1)*[0 0 1 1];
ydat1 = l1*[0 0 sin(q1)*1 sin(q1)*1];
link1 = patch(xdat1,ydat1,'r');

%Create link2:
width2 = l2*0.1;
xdat2 = (0.5*width1)*[-1 1 1 -1]+l1*cos(q1)*[1 1 1 1]+l2*cos(q1+q2)*[0 0 1 1];
ydat2 = l1*[sin(q1)*1 sin(q1)*1 sin(q1)*1 sin(q1)*1]+l2*[0 0 sin(q1+q2)*1 sin(q1+q2)*1];
link2 = patch(xdat2,ydat2,'b');
axis([-3.5 3.5 -3.6 3.6]);

%Dots for the hinges:
h1 = plot(0,0,'.k','MarkerSize',40); %First link anchor
h2 = plot(0,0,'.k','MarkerSize',40); %link1 -> link2 hinge

%joint value meters on screen
q11 = string(q1);
q11 = q11(1:end);
q22 = string(q2);
q22 = q22(1:end);
text(1.5,-2.6,q11,'FontSize',22,'Color', 'r');
text(0.1,-2.6,'q1 = ','FontSize',22,'Color', 'r');
text(1.5,-3.2,q22,'FontSize',22,'Color', 'b');
text(0.1,-3.2,'q2 = ','FontSize',22,'Color', 'b');
tau = [0, 0];

%mark the end point
plot(x0,y0,'xb','MarkerSize',30);

hold off

%Make the whole window big for handy viewing:
set(f, 'units', 'inches', 'position', [5 5 5 5])
set(f,'Color',[1,1,1]);