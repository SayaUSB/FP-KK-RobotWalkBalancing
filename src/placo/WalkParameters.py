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
    
    def __init__(self, parameters: Dict=None):
        self.params = placo.HumanoidParameters()
        self.parameters = parameters
        self._set_default_parameters()
        self._define_tunable_bounds()
        
    def _set_default_parameters(self):
        """Set default parameters"""
        
        if not self.parameters:
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
            self.params.walk_max_dx_forward = 0.36
            self.params.walk_max_dx_backward = 0.03
        else:
            # Timing parameters
            self.params.single_support_duration = self.parameters['single_support_duration']
            self.params.single_support_timesteps = self.parameters['single_support_timesteps']
            self.params.double_support_ratio = self.parameters['double_support_ratio']
            self.params.startend_double_support_ratio = self.parameters['startend_double_support_ratio']
            self.params.planned_timesteps = self.parameters['planned_timesteps']

            # Posture parameters
            self.params.walk_com_height = self.parameters['walk_com_height']
            self.params.walk_foot_height = self.parameters['walk_foot_height']
            self.params.walk_trunk_pitch = self.parameters['walk_trunk_pitch']
            self.params.walk_foot_rise_ratio = self.parameters['walk_foot_rise_ratio']

            # Feet parameters
            self.params.foot_length = self.parameters['foot_length']
            self.params.foot_width = self.parameters['foot_width']
            self.params.feet_spacing = self.parameters['feet_spacing']
            self.params.walk_max_dtheta = self.parameters['walk_max_dtheta']
            self.params.walk_max_dy = self.parameters['feet_spacing']
            self.params.foot_zmp_target_x = self.parameters['foot_zmp_target_x']
            self.params.foot_zmp_target_y = self.parameters['foot_zmp_target_y']
            
            # Limit parameters
            self.params.walk_max_dtheta = self.parameters['walk_max_dtheta']
            self.params.walk_max_dy = self.parameters['walk_max_dy']
            self.params.walk_max_dx_forward = self.parameters['walk_max_dx_forward']
            self.params.walk_max_dx_backward = self.parameters['walk_max_dx_backward']
        
    def _define_tunable_bounds(self):
        """Define parameter bounds untuk optimization"""
        self.tunable_params = {
            # Timing parameters
            'single_support_duration': ParameterBounds(
                'single_support_duration', 0.2, 0.6, 0.38, float
            ),
            'single_support_timesteps': ParameterBounds(
                'single_support_timesteps', 1, 20, 10, int
            ),
            'double_support_ratio': ParameterBounds(
                'double_support_ratio', 0.0, 0.3, 0.0, float
            ),
            'startend_double_support_ratio': ParameterBounds(
                'startend_double_support_ratio', 1.0, 2.0, 1.5, float
            ),
            'planned_timesteps': ParameterBounds(
                'planned_timesteps', 10, 50, 48, int
            ),
            
            # Posture parameters
            'walk_com_height': ParameterBounds(
                'walk_com_height', 0.25, 0.40, 0.32, float
            ),
            'walk_foot_height': ParameterBounds(
                'walk_foot_height', 0.02, 0.08, 0.04, float
            ),
            'walk_trunk_pitch': ParameterBounds(
                'walk_trunk_pitch', 0.0, 0.3, 0.15, float
            ),
            'walk_foot_rise_ratio': ParameterBounds(
                'walk_foot_rise_ratio', 0.1, 0.4, 0.2, float
            ),
            
            # Feet parameters
            'foot_length': ParameterBounds(
                'feet_spacing', 0, 0.3, 0.1576, float
            ),
            'foot_width': ParameterBounds(
                'foot_width', 0, 0.3, 0.092, float
            ),
            'feet_spacing': ParameterBounds(
                'feet_spacing', 0.10, 0.15, 0.122, float
            ),
            'zmp_margin': ParameterBounds(
                'zmp_margin', 0.01, 0.04, 0.02, float
            ),
            'foot_zmp_target_x': ParameterBounds(
                'foot_zmp_target_x', -0.02, 0.02, 0.0, float
            ),
            'foot_zmp_target_y': ParameterBounds(
                'foot_zmp_target_y', -0.02, 0.02, 0.0, float
            ),
            
            # Limit parameters
            'walk_max_dtheta': ParameterBounds(
                'walk_max_dtheta', 0.5, 1.5, 1.0,float
            ),
            'walk_max_dy': ParameterBounds(
                'walk_max_dy', 0.02, 0.08, 0.04, float
            ),
            'walk_max_dx_forward': ParameterBounds(
                'walk_max_dx_forward', 0.5, 1.5, 1.0, float
            ),
            'walk_max_dx_backward': ParameterBounds(
                'walk_max_dx_backward', 0.02, 0.06, 0.03, float
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