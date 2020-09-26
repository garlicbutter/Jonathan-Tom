function Plotter_Trajectory(p)

%set the lines
p12_x = linspace(p.p1.x,p.p2.x,1000);
p12_y = linspace(p.p1.y,p.p2.y,1000);
p23_x = linspace(p.p2.x,p.p3.x,1000);
p23_y = linspace(p.p2.y,p.p3.y,1000);
p34_x = linspace(p.p3.x,p.p4.x,1000);
p34_y = linspace(p.p3.y,p.p4.y,1000);
p41_x = linspace(p.p4.x,p.p1.x,1000);
p41_y = linspace(p.p4.y,p.p1.y,1000);

%Name the whole window and define the mouse callback function
f = figure;
set(f,'WindowButtonMotionFcn','','WindowButtonDownFcn',@ClickDown,'WindowButtonUpFcn',@ClickUp,'KeyPressFc',@KeyPress);

%%%%%%%% 1st Subplot -- the pendulum animation %%%%%%%
figData.simArea = subplot(1,1,1); %Eliminated other subplots, but left this for syntax consistency.
axis equal
hold on

plot(p12_x,p12_y,'r--');
plot(p23_x,p23_y,'r--');
plot(p34_x,p34_y,'r--');
plot(p41_x,p41_y,'r--');

end 