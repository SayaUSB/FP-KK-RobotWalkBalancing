import json
import placo
from typing import Dict
from dataclasses import dataclass

@dataclass
class ParameterBounds:
    """Bounds untuk setiap parameter yang akan di-tune"""
    name: str
    min_value: float
    max_value: float
    default_value: float
    
    def __post_init__(self):
        assert self.min_value <= self.default_value <= self.max_value, \
            f"Default value {self.default_value} not in bounds [{self.min_value}, {self.max_value}]"
    
    def to_dict(self):
        return {
            'name': self.name,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'default_value': self.default_value
        }


class WalkParameters:
    """Parameters for robot gait with tuning support"""
    
    def __init__(self):
        self.params = placo.HumanoidParameters()
        self._set_default_parameters()
        self._define_tunable_bounds()
        
    def _set_default_parameters(self):
        """Set default parameters"""
        # Timing parameters
        self.params.single_support_duration = 0.6
        self.params.single_support_timesteps = 10
        self.params.double_support_ratio = 0.0006008777616272946
        self.params.startend_double_support_ratio = 1.5886696014263284
        self.params.planned_timesteps = 48
        
        # Posture parameters
        self.params.walk_com_height = 0.32236855497734307
        self.params.walk_foot_height = 0.0373010486746116
        self.params.walk_trunk_pitch = 0.1515944847805595
        self.params.walk_foot_rise_ratio = 0.20113722567696113
        
        # Feet parameters
        self.params.foot_length = 0.1576
        self.params.foot_width = 0.092
        self.params.feet_spacing = 0.12185145577350875
        self.params.zmp_margin = 0.01998778155747486
        self.params.foot_zmp_target_x = -0.001560746760865157
        self.params.foot_zmp_target_y = -0.00033707127920877284
        
        # Limit parameters
        self.params.walk_max_dtheta = 1.0096951500288134
        self.params.walk_max_dy = 0.04232503853606945
        self.params.walk_max_dx_forward = 1.0044747278638186
        self.params.walk_max_dx_backward = 0.02960790599310682
        
    def _define_tunable_bounds(self):
        """Define parameter bounds untuk optimization"""
        self.tunable_params = {
            # Timing parameters
            'single_support_duration': ParameterBounds(
                'single_support_duration', 0.2, 0.6, 0.38
            ),
            'double_support_ratio': ParameterBounds(
                'double_support_ratio', 0.0, 0.3, 0.0
            ),
            'startend_double_support_ratio': ParameterBounds(
                'startend_double_support_ratio', 1.0, 2.0, 1.5
            ),
            
            # Posture parameters
            'walk_com_height': ParameterBounds(
                'walk_com_height', 0.25, 0.40, 0.32
            ),
            'walk_foot_height': ParameterBounds(
                'walk_foot_height', 0.02, 0.08, 0.04
            ),
            'walk_trunk_pitch': ParameterBounds(
                'walk_trunk_pitch', 0.0, 0.3, 0.15
            ),
            'walk_foot_rise_ratio': ParameterBounds(
                'walk_foot_rise_ratio', 0.1, 0.4, 0.2
            ),
            
            # Feet parameters
            'feet_spacing': ParameterBounds(
                'feet_spacing', 0.10, 0.15, 0.122
            ),
            'zmp_margin': ParameterBounds(
                'zmp_margin', 0.01, 0.04, 0.02
            ),
            'foot_zmp_target_x': ParameterBounds(
                'foot_zmp_target_x', -0.02, 0.02, 0.0
            ),
            'foot_zmp_target_y': ParameterBounds(
                'foot_zmp_target_y', -0.02, 0.02, 0.0
            ),
            
            # Limit parameters
            'walk_max_dtheta': ParameterBounds(
                'walk_max_dtheta', 0.5, 1.5, 1.0
            ),
            'walk_max_dy': ParameterBounds(
                'walk_max_dy', 0.02, 0.08, 0.04
            ),
            'walk_max_dx_forward': ParameterBounds(
                'walk_max_dx_forward', 0.5, 1.5, 1.0
            ),
            'walk_max_dx_backward': ParameterBounds(
                'walk_max_dx_backward', 0.02, 0.06, 0.03
            ),
        }
        
    def get_parameters(self):
        """Return parameter object"""
        return self.params
        
    def set_parameter(self, param_name: str, value: float):
        """Set single parameter value"""
        if hasattr(self.params, param_name):
            setattr(self.params, param_name, value)
        else:
            raise ValueError(f"Parameter {param_name} not found")
            
    def set_parameters_from_dict(self, param_dict: Dict[str, float]):
        """Set multiple parameters from dictionary"""
        for name, value in param_dict.items():
            self.set_parameter(name, value)
            
    def get_parameter_dict(self) -> Dict[str, float]:
        """Get current parameters as dictionary"""
        return {
            name: getattr(self.params, name)
            for name in self.tunable_params.keys()
        }
        
    def get_tunable_bounds(self) -> Dict[str, ParameterBounds]:
        """Get bounds for tunable parameters"""
        return self.tunable_params
    
    def save_to_file(self, filename: str):
        """Save parameters to JSON file"""
        data = {
            'parameters': self.get_parameter_dict(),
            'bounds': {
                name: bounds.to_dict() 
                for name, bounds in self.tunable_params.items()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load parameters from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        self.set_parameters_from_dict(data['parameters'])