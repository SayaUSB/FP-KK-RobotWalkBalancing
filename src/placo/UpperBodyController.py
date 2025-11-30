import numpy as np

class UpperBodyController:
    """Control the position of the upper body of the robot"""
    
    def __init__(self, solver):
        self.solver = solver
        self.joints_task = None
        self._setup_joints_task()
        
    def _setup_joints_task(self):
        """Setup task for upper body joints"""
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
        """Update certain joint positions"""
        self.joints_task.set_joints(joint_dict)