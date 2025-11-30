import time
import numpy as np
from placo_utils.visualization import (
    robot_viz,
    frame_viz,
    line_viz,
    footsteps_viz,
)

class SimulationManager:
    """Manager for visualization and simulation"""
    
    def __init__(self, controller, use_pybullet=False, use_meshcat=False):
        self.controller = controller
        self.use_pybullet = use_pybullet
        self.use_meshcat = use_meshcat
        
        self.sim = None
        self.viz = None
        self.last_display = time.time()
        self.initial_delay = -2.0 if (use_pybullet or use_meshcat) else 0.0
        self.start_t = time.time()
        
        self._setup_visualization()
        
    def _setup_visualization(self):
        """Setup PyBullet or MeshCat"""
        if self.use_pybullet:
            import pybullet as p
            from onshape_to_robot.simulation import Simulation
            self.sim = Simulation(
                self.controller.model_filename, 
                realTime=True, 
                dt=self.controller.dt
            )
            
        elif self.use_meshcat:
            self.viz = robot_viz(self.controller.robot)
            footsteps_viz(self.controller.trajectory.get_supports())
            
    def get_com_position(self):
        """Get CoM position from simulation"""
        if self.use_pybullet:
            import pybullet as p
            com_pos, _ = p.getBasePositionAndOrientation(1)
            return com_pos[2]
        else:
            # For meshcat or without simulation, take from robot
            return self.controller.robot.com_world()[2]
            
    def update_pybullet(self, t):
        """Update PyBullet simulation"""
        if not self.use_pybullet:
            return
            
        if t < -2:
            T_left_origin = self.sim.transformation("origin", "left_foot_frame")
            T_world_left = self.sim.poseToMatrix(([0.0, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]))
            T_world_origin = T_world_left @ T_left_origin
            self.sim.setRobotPose(*self.sim.matrixToPose(T_world_origin))
            
        joints = {
            joint: self.controller.robot.get_joint(joint) 
            for joint in self.sim.getJoints()
        }
        self.sim.setJoints(joints)
        self.sim.tick()
        
    def update_meshcat(self, supports=None):
        """Update MeshCat visualization"""
        if not self.use_meshcat:
            return
            
        if time.time() - self.last_display > 0.03:
            self.last_display = time.time()
            self.viz.display(self.controller.robot.state.q)
            
            t = self.controller.t
            traj = self.controller.trajectory
            
            frame_viz("left_foot_target", traj.get_T_world_left(t))
            frame_viz("right_foot_target", traj.get_T_world_right(t))
            
            T_world_trunk = np.eye(4)
            T_world_trunk[:3, :3] = traj.get_R_world_trunk(t)
            T_world_trunk[:3, 3] = traj.get_p_world_CoM(t)
            frame_viz("trunk_target", T_world_trunk)
            
            if supports is not None:
                footsteps_viz(supports)
                
                # Draw CoM trajectory
                coms = [
                    [*traj.get_p_world_CoM(t_i)[:2], 0.0]
                    for t_i in np.linspace(traj.t_start, traj.t_end, 100)
                ]
                line_viz("CoM_trajectory", np.array(coms), 0xFFAA00)
                
    def spin_lock(self, t):
        """Wait until next event"""
        while time.time() + self.initial_delay < self.start_t + t:
            time.sleep(1e-3)