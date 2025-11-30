import pinocchio
import time
import placo
import argparse
import numpy as np
import warnings
import logging
from placo_utils.visualization import (
    robot_viz,
    frame_viz,
    line_viz,
    footsteps_viz,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class FallDetector:
    """Mendeteksi kondisi jatuh pada robot"""
    
    def __init__(self, com_z_threshold=0.3, trunk_angle_threshold=0.5, com_drop_threshold=0.05):
        self.com_z_threshold = com_z_threshold
        self.trunk_angle_threshold = trunk_angle_threshold
        self.com_drop_threshold = com_drop_threshold
        self.logger = logging.getLogger("fall_detection")
        
    def check_fall(self, com_z, roll, pitch):
        """
        Cek apakah robot jatuh berdasarkan CoM dan sudut trunk
        
        Args:
            com_z: Ketinggian center of mass
            roll: Sudut roll trunk (rad)
            pitch: Sudut pitch trunk (rad)
            
        Returns:
            bool: True jika terdeteksi jatuh
        """
        if com_z < self.com_z_threshold:
            self.logger.warning(f"Fall detected: CoM too low ({com_z:.3f} < {self.com_z_threshold})")
            return True
            
        if abs(roll) > self.trunk_angle_threshold or abs(pitch) > self.trunk_angle_threshold:
            self.logger.warning(f"Fall detected: Trunk angle too large (roll={roll:.3f}, pitch={pitch:.3f})")
            return True
            
        return False


class WalkParameters:
    """Parameter untuk gaya berjalan robot"""
    
    def __init__(self):
        self.params = placo.HumanoidParameters()
        self._set_default_parameters()
        
    def _set_default_parameters(self):
        """Set parameter default"""
        # Timing parameters
        self.params.single_support_duration = 0.38
        self.params.single_support_timesteps = 10
        self.params.double_support_ratio = 0.0
        self.params.startend_double_support_ratio = 1.5
        self.params.planned_timesteps = 48
        
        # Posture parameters
        self.params.walk_com_height = 0.32
        self.params.walk_foot_height = 0.04
        self.params.walk_trunk_pitch = 0.15
        self.params.walk_foot_rise_ratio = 0.2
        
        # Feet parameters
        self.params.foot_length = 0.1576
        self.params.foot_width = 0.092
        self.params.feet_spacing = 0.122
        self.params.zmp_margin = 0.02
        self.params.foot_zmp_target_x = 0.0
        self.params.foot_zmp_target_y = 0.0
        
        # Limit parameters
        self.params.walk_max_dtheta = 1
        self.params.walk_max_dy = 0.04
        self.params.walk_max_dx_forward = 1
        self.params.walk_max_dx_backward = 0.03
        
    def get_parameters(self):
        """Return parameter object"""
        return self.params


class UpperBodyController:
    """Kontrol posisi upper body robot"""
    
    def __init__(self, solver):
        self.solver = solver
        self.joints_task = None
        self._setup_joints_task()
        
    def _setup_joints_task(self):
        """Setup task untuk joint upper body"""
        elbow = -50 * np.pi / 180
        shoulder_roll = 0 * np.pi / 180
        shoulder_pitch = 20 * np.pi / 180
        
        self.joints_task = self.solver.add_joints_task()
        self.joints_task.set_joints({
            "left_shoulder_roll": shoulder_roll,
            "left_shoulder_pitch": shoulder_pitch,
            "left_elbow": elbow,
            "right_shoulder_roll": -shoulder_roll,
            "right_shoulder_pitch": shoulder_pitch,
            "right_elbow": elbow,
            "head_pitch": 0.0,
            "head_yaw": 0.0,
        })
        self.joints_task.configure("joints", "soft", 1.0)
        
    def update_joint_position(self, joint_dict):
        """Update posisi joint tertentu"""
        self.joints_task.set_joints(joint_dict)


class FootstepPlanner:
    """Perencanaan footstep untuk walking"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.planner = placo.FootstepsPlannerRepetitive(parameters)
        
    def configure_walk(self, d_x=0.1, d_y=0.0, d_theta=0.0, nb_steps=10):
        """
        Konfigurasi parameter berjalan
        
        Args:
            d_x: Jarak langkah forward/backward (m)
            d_y: Jarak langkah lateral (m)
            d_theta: Rotasi per langkah (rad)
            nb_steps: Jumlah langkah
        """
        self.planner.configure(d_x, d_y, d_theta, nb_steps)
        
    def plan_footsteps(self, robot, start_side=placo.HumanoidRobot_Side.left):
        """
        Plan footsteps dari posisi robot saat ini
        
        Args:
            robot: Object HumanoidRobot
            start_side: Sisi kaki yang memulai langkah
            
        Returns:
            tuple: (footsteps, supports)
        """
        T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
        T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
        
        footsteps = self.planner.plan(start_side, T_world_left, T_world_right)
        supports = placo.FootstepsPlanner.make_supports(
            footsteps, 0.0, True, 
            self.parameters.has_double_support(), True
        )
        
        return footsteps, supports


class HumanoidWalkController:
    """Controller utama untuk walking humanoid robot"""
    
    def __init__(self, model_filename, dt=0.005, replan_dt=0.1):
        self.model_filename = model_filename
        self.dt = dt
        self.replan_dt = replan_dt
        
        # Initialize components
        self.robot = placo.HumanoidRobot(model_filename)
        self.walk_params = WalkParameters()
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
        """Tempatkan robot pada posisi awal"""
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
        Plan trajectory awal untuk walking
        
        Args:
            d_x, d_y, d_theta: Parameter langkah
            nb_steps: Jumlah langkah
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
        Dapatkan orientasi trunk dalam euler angles
        
        Returns:
            tuple: (roll, pitch, yaw) dalam radian
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
        Cek stabilitas robot
        
        Args:
            com_z: Ketinggian CoM
            
        Returns:
            bool: True jika stabil, False jika jatuh
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
        """Replan trajectory jika sudah waktunya"""
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
        """Satu iterasi control loop"""
        self.update_trajectory()
        self.solve_ik()
        self.replan_if_needed()
        self.t += self.dt


class SimulationManager:
    """Manager untuk visualisasi dan simulasi"""
    
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
        """Setup PyBullet atau MeshCat"""
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
        """Dapatkan posisi CoM dari simulasi"""
        if self.use_pybullet:
            import pybullet as p
            com_pos, _ = p.getBasePositionAndOrientation(1)
            return com_pos[2]
        else:
            # Untuk meshcat atau tanpa simulasi, ambil dari robot
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
        """Wait sampai waktu berikutnya"""
        while time.time() + self.initial_delay < self.start_t + t:
            time.sleep(1e-3)


def main():
    parser = argparse.ArgumentParser(description="Humanoid walking controller")
    parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
    parser.add_argument("-m", "--meshcat", action="store_true", help="MeshCat visualization")
    args = parser.parse_args()
    
    if not args.pybullet and not args.meshcat:
        print("No visualization selected, use either -p or -m")
        return
        
    # Initialize controller
    model_filename = "model/sigmaban/robot.urdf"
    controller = HumanoidWalkController(model_filename)
    
    # Setup robot
    controller.initialize_robot_pose()
    controller.plan_initial_trajectory(d_x=0.1, d_y=0.0, d_theta=0.0, nb_steps=10)
    
    # Setup simulation/visualization
    sim_manager = SimulationManager(
        controller,
        use_pybullet=args.pybullet,
        use_meshcat=args.meshcat
    )
    
    # Main control loop
    while True:
        # Get CoM position
        com_z = sim_manager.get_com_position()
        
        # Check stability
        if not controller.check_stability(com_z):
            time.sleep(1)
            break
            
        # Execute one control step
        controller.step()
        
        # Update visualization
        supports = controller.replan_if_needed()
        sim_manager.update_pybullet(controller.t)
        sim_manager.update_meshcat(supports)
        
        # Timing control
        sim_manager.spin_lock(controller.t)


if __name__ == "__main__":
    main()