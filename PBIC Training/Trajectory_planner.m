function [traj] =Trajectory_planner(p)
%Return the [x,y] array for the given trajectory points
traj = [];
traj_follow_rate = 100;
for traj_counter = 1: length(p.traj)-1
    traj_lin_max = ceil((pdist(p.traj(traj_counter:traj_counter+1,:))*traj_follow_rate));
    temp_array1 = linspace(p.traj(traj_counter,1),p.traj(traj_counter+1,1),traj_lin_max);
    temp_array2 = linspace(p.traj(traj_counter,2),p.traj(traj_counter+1,2),traj_lin_max); 
    for traj_lin_counter = 1: traj_lin_max
            traj = [traj; temp_array1(traj_lin_counter), temp_array2(traj_lin_counter)];
    end
end


end

