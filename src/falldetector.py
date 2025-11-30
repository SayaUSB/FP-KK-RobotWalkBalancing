import logging
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