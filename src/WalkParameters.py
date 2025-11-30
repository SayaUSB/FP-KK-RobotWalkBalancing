import placo

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