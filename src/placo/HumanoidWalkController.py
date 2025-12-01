import placo
import logging
import numpy as np
from .WalkParameters import WalkParameters
from .UpperBodyController import UpperBodyController
from .FootstepPlanner import FootstepPlanner
from .FallDetector import FallDetector

class HumanoidWalkController:
    """Main controller for walking humanoid robot"""
    
    def __init__(self, model_filename, dt=0.005, replan_dt=0.1, parameters: dict=None):
        self.model_filename = model_filename
        self.dt = dt
        self.replan_dt = replan_dt
        
        # Initialize components
        self.robot = placo.HumanoidRobot(model_filename)
        self.walk_params = WalkParameters(parameters)
        self.parameters = self.walk_params.get_parameters()
        
        # Setup solver
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.enable_velocity_limits(True)
        self.solver.dt = dt
        
        # Setup tasks and controllers
        self.tasks = placo.WalkTasks()
        self.tasks.initialize_tasks(self.solver, self.robot)
        self.upper_body = UpperBodyController(self.solver)
        
        # Setup footstep planner
        self.footstep_planner = FootstepPlanner(self.parameters)
        self.walk = placo.WalkPatternGenerator(self.robot, self.parameters)
        
        # Fall detector
        self.fall_detector = FallDetector()
        
        # State variables
        self.trajectory = None
        self.last_replan = 0
        self.t = 0
        
        self.logger = logging.getLogger("walk_controller")

    def initialize_robot_pose(self):
        """Place the robot in the starting position"""
        self.logger.info("Placing the robot in the initial position...")
        self.tasks.reach_initial_pose(
            np.eye(4),
            self.parameters.feet_spacing,
            self.parameters.walk_com_height,
            self.parameters.walk_trunk_pitch,
        )
        self.logger.info("Initial position reached")
        
    def plan_initial_trajectory(self, d_x=0.1, d_y=0.0, d_theta=0.0, nb_steps=10):
        """
        Initial trajectory plan for walking
        
        Args:
            d_x, d_y, d_theta: Step parameters
            nb_steps: Number of steps
        """
        self.footstep_planner.configure_walk(d_x, d_y, d_theta, nb_steps)
        footsteps, supports = self.footstep_planner.plan_footsteps(self.robot)
        
        self.trajectory = self.walk.plan(
            supports, 
            self.robot.com_world(), 
            0.0
        )
        
        return self.trajectory
        
    def get_trunk_orientation(self):
        """
        Get the trunk orientation in euler angles
        
        Returns:
            tuple: (roll, pitch, yaw) in radian
        """
        R_trunk = None
        try:
            T_trunk = self.robot.get_T_world_trunk()
            R_trunk = T_trunk[:3, :3]
        except Exception:
            try:
                R_trunk = self.robot.get_R_world_trunk()
            except Exception:
                try:
                    R_trunk = self.trajectory.get_R_world_trunk(self.t)
                except Exception:
                    R_trunk = np.eye(3)
                    
        return self._rot_to_rpy(R_trunk)
    
    @staticmethod
    def _rot_to_rpy(R):
        """Convert rotation matrix to roll, pitch, yaw"""
        r = np.arctan2(R[2, 1], R[2, 2])
        p = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        y = np.arctan2(R[1, 0], R[0, 0])
        return r, p, y
        
    def check_stability(self, com_z):
        """
        Check robot stability
        
        Args:
            com_z: Height of CoM
            
        Returns:
            bool: True if stable, False if falls
        """
        roll, pitch, yaw = self.get_trunk_orientation()
        
        if self.fall_detector.check_fall(com_z, roll, pitch):
            print("=== FALL DETECTED ===")
            return False
            
        return True
        
    def update_trajectory(self):
        """Update trajectory dari planned trajectory"""
        self.tasks.update_tasks_from_trajectory(self.trajectory, self.t)
        
    def solve_ik(self):
        """Solve inverse kinematics"""
        self.robot.update_kinematics()
        qd_sol = self.solver.solve(True)
        
        # Ensure robot on floor
        if not self.trajectory.support_is_both(self.t):
            self.robot.update_support_side(str(self.trajectory.support_side(self.t)))
            self.robot.ensure_on_floor()
            
        return qd_sol
        
    def replan_if_needed(self):
        """Replan trajectory when it's time to"""
        if (self.t - self.last_replan > self.replan_dt and 
            self.walk.can_replan_supports(self.trajectory, self.t)):
            
            supports = self.walk.replan_supports(
                self.footstep_planner.planner,
                self.trajectory,
                self.t,
                self.last_replan
            )
            
            self.last_replan = self.t
            self.trajectory = self.walk.replan(supports, self.trajectory, self.t)
            
            return supports
        return None
        
    def step(self):
        """One iteration of the control loop"""
        self.update_trajectory()
        self.solve_ik()
        self.replan_if_needed()
        self.t += self.dt