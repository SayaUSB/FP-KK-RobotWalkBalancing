import numpy as np
import logging
from typing import Dict

logger = logging.getLogger("fitness_evaluator")


class WalkFitnessEvaluator:
    """Evaluator untuk fitness walking parameters"""
    
    def __init__(
        self,
        controller_class,
        model_filename: str,
        simulation_duration: float = 5.0,
        target_distance: float = 0.5,
        weights: Dict[str, float] = None,
    ):
        """
        Args:
            controller_class: HumanoidWalkController class
            model_filename: Path ke URDF file
            simulation_duration: Duration simulasi untuk evaluasi (seconds)
            target_distance: Target jarak yang ingin dicapai (meters)
            weights: Dictionary bobot untuk setiap komponen fitness
        """
        self.controller_class = controller_class
        self.model_filename = model_filename
        self.simulation_duration = simulation_duration
        self.target_distance = target_distance
        
        # Default weights
        self.weights = weights or {
            'fall_penalty': 1000.0,
            'distance_error': 100.0,
            'instability': 50.0,
            'parameter_smoothness': 10.0,
        }
        
    def evaluate(self, param_dict: Dict[str, float]) -> float:
        """
        Evaluate fitness untuk set parameters tertentu
        
        Returns:
            float: Fitness score (lower is better)
        """
        try:
            # Create controller dengan parameters baru
            controller = self.controller_class(self.model_filename)
            
            # Set parameters
            controller.walk_params.set_parameters_from_dict(param_dict)
            
            # Initialize robot
            controller.initialize_robot_pose()
            controller.plan_initial_trajectory(d_x=0.1, d_y=0.0, d_theta=0.0, nb_steps=10)
            
            # Run simulation
            initial_pos = controller.robot.com_world()[:2].copy()
            fall_detected = False
            unstable_count = 0
            total_steps = 0
            
            t = 0
            dt = controller.dt
            
            while t < self.simulation_duration:
                # Get CoM
                com_world = controller.robot.com_world()
                com_z = com_world[2]
                
                # Check stability
                roll, pitch, yaw = controller.get_trunk_orientation()
                
                # Detect fall or instability
                if com_z < 0.25 or abs(roll) > 0.6 or abs(pitch) > 0.6:
                    fall_detected = True
                    break
                    
                if com_z < 0.28 or abs(roll) > 0.4 or abs(pitch) > 0.4:
                    unstable_count += 1
                    
                # Step simulation
                controller.step()
                t += dt
                total_steps += 1
                
            # Calculate fitness components
            final_pos = controller.robot.com_world()[:2]
            distance_traveled = np.linalg.norm(final_pos - initial_pos)
            
            # Fitness = weighted sum of objectives (lower is better)
            fitness = 0.0
            
            # 1. Penalize fall heavily
            if fall_detected:
                fitness += self.weights['fall_penalty']
                
            # 2. Distance error
            distance_error = abs(self.target_distance - distance_traveled)
            fitness += distance_error * self.weights['distance_error']
            
            # 3. Penalize instability
            instability_ratio = unstable_count / max(total_steps, 1)
            fitness += instability_ratio * self.weights['instability']
            
            # 4. Penalize extreme parameters (for smoothness)
            param_penalty = self._calculate_parameter_penalty(param_dict, controller)
            fitness += param_penalty * self.weights['parameter_smoothness']
            
            return fitness
            
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return 10000.0  # Very high penalty for failed evaluation
    
    def _calculate_parameter_penalty(self, param_dict, controller):
        """Calculate penalty for extreme parameter values"""
        param_penalty = 0.0
        for name, value in param_dict.items():
            bounds = controller.walk_params.tunable_params[name]
            range_size = bounds.max_value - bounds.min_value
            normalized_deviation = abs(value - bounds.default_value) / range_size
            param_penalty += normalized_deviation
        return param_penalty