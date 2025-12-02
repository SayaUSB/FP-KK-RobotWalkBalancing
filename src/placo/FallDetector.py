import logging
logging.basicConfig(level=logging.INFO)

class FallDetector:
    """Detect if the robot falls"""
    
    def __init__(self, com_z_threshold=0.25, trunk_angle_threshold=0.5, com_drop_threshold=0.05):
        self.com_z_threshold = com_z_threshold
        self.trunk_angle_threshold = trunk_angle_threshold
        self.com_drop_threshold = com_drop_threshold
        self.logger = logging.getLogger("fall_detection")
        
    def check_fall(self, com_z, roll, pitch):
        """
        Check whether the robot falls based on the CoM and trunk angle
        
        Args:
            com_z: Height of center of mass
            roll: Trunk roll angle (rad)
            pitch: Trunk roll angle (rad)
            
        Returns:
            bool: True if falls
        """
        if com_z < self.com_z_threshold:
            self.logger.warning(f"Fall detected: CoM too low ({com_z:.3f} < {self.com_z_threshold})")
            return True
            
        if abs(roll) > self.trunk_angle_threshold or abs(pitch) > self.trunk_angle_threshold:
            self.logger.warning(f"Fall detected: Trunk angle too large (roll={roll:.3f}, pitch={pitch:.3f})")
            return True
            
        return False