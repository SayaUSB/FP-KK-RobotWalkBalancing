import numpy as np
import logging
from typing import Dict, Callable, Optional, Tuple
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pso_optimizer")


class Particle:
    """Single particle dalam PSO swarm"""
    
    def __init__(self, bounds: Dict[str, 'ParameterBounds']):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)
        
        # Initialize position randomly within bounds
        self.position = np.array([
            np.random.uniform(bounds[name].min_value, bounds[name].max_value)
            for name in self.param_names
        ])
        
        # Initialize velocity
        velocity_range = np.array([
            (bounds[name].max_value - bounds[name].min_value) * 0.1
            for name in self.param_names
        ])
        self.velocity = np.random.uniform(-velocity_range, velocity_range)
        
        # Personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.current_fitness = float('inf')
        
    def to_dict(self) -> Dict[str, float]:
        """Convert position to parameter dictionary"""
        return {
            name: self.position[i]
            for i, name in enumerate(self.param_names)
        }
        
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        """Update particle velocity"""
        r1 = np.random.random(self.n_dims)
        r2 = np.random.random(self.n_dims)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
    def update_position(self):
        """Update particle position and apply bounds"""
        self.position += self.velocity
        
        # Apply bounds
        for i, name in enumerate(self.param_names):
            bounds = self.bounds[name]
            self.position[i] = np.clip(
                self.position[i],
                bounds.min_value,
                bounds.max_value
            )
            
    def update_best(self, fitness: float):
        """Update personal best if fitness improved"""
        self.current_fitness = fitness
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            return True
        return False


class PSOOptimizer:
    """Particle Swarm Optimization algorithm"""
    
    def __init__(
        self,
        bounds: Dict[str, 'ParameterBounds'],
        fitness_function: Callable[[Dict[str, float]], float],
        n_particles: int = 20,
        max_iterations: int = 50,
        w: float = 0.7,  # inertia weight
        c1: float = 1.5,  # cognitive parameter
        c2: float = 1.5,  # social parameter
    ):
        """
        Args:
            bounds: Dictionary of parameter bounds
            fitness_function: Function yang menerima dict parameters dan return fitness score (lower is better)
            n_particles: Jumlah particles dalam swarm
            max_iterations: Maximum iterasi optimization
            w: Inertia weight
            c1: Cognitive parameter (personal best influence)
            c2: Social parameter (global best influence)
        """
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.particles = []
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_params = None
        
        # History
        self.fitness_history = []
        self.best_fitness_history = []
        self.iteration_best_params = []
        
    def initialize_swarm(self):
        """Initialize particle swarm"""
        logger.info(f"Initializing swarm with {self.n_particles} particles")
        self.particles = [Particle(self.bounds) for _ in range(self.n_particles)]
        
    def evaluate_particle(self, particle: Particle) -> float:
        """Evaluate fitness for a particle"""
        param_dict = particle.to_dict()
        fitness = self.fitness_function(param_dict)
        return fitness
        
    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Run PSO optimization
        
        Args:
            verbose: Print progress
            callback: Function called after each iteration with (iteration, best_fitness, best_params)
            
        Returns:
            tuple: (best_parameters_dict, best_fitness)
        """
        self.initialize_swarm()
        
        for iteration in range(self.max_iterations):
            iteration_fitnesses = []
            
            # Evaluate all particles
            for particle in self.particles:
                fitness = self.evaluate_particle(particle)
                iteration_fitnesses.append(fitness)
                
                # Update personal best
                particle.update_best(fitness)
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
                    self.global_best_params = particle.to_dict()
                    
                    if verbose:
                        logger.info(
                            f"Iteration {iteration + 1}/{self.max_iterations}: "
                            f"New best fitness = {fitness:.6f}"
                        )
                        
            # Record history
            avg_fitness = np.mean(iteration_fitnesses)
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(self.global_best_fitness)
            self.iteration_best_params.append(self.global_best_params.copy())
            
            if verbose and iteration % 5 == 0:
                logger.info(
                    f"Iteration {iteration + 1}/{self.max_iterations}: "
                    f"Avg fitness = {avg_fitness:.6f}, "
                    f"Best fitness = {self.global_best_fitness:.6f}"
                )
                
            # Callback
            if callback:
                callback(iteration, self.global_best_fitness, self.global_best_params)
                
            # Update all particles
            for particle in self.particles:
                particle.update_velocity(
                    self.global_best_position,
                    self.w,
                    self.c1,
                    self.c2
                )
                particle.update_position()
                
        if verbose:
            logger.info(f"\nOptimization completed!")
            logger.info(f"Best fitness: {self.global_best_fitness:.6f}")
            logger.info(f"Best parameters:")
            for name, value in self.global_best_params.items():
                logger.info(f"  {name}: {value:.6f}")
                
        return self.global_best_params, self.global_best_fitness
        
    def get_optimization_history(self) -> Dict:
        """Get optimization history"""
        return {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'iteration_best_params': self.iteration_best_params,
            'global_best_params': self.global_best_params,
            'global_best_fitness': self.global_best_fitness,
        }
        
    def save_results(self, filename: str):
        """Save optimization results to JSON file"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'pso_config': {
                'n_particles': self.n_particles,
                'max_iterations': self.max_iterations,
                'w': self.w,
                'c1': self.c1,
                'c2': self.c2,
            },
            'best_parameters': self.global_best_params,
            'best_fitness': float(self.global_best_fitness),
            'fitness_history': [float(f) for f in self.fitness_history],
            'best_fitness_history': [float(f) for f in self.best_fitness_history],
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {filename}")
        
    @staticmethod
    def load_results(filename: str) -> Dict:
        """Load optimization results from JSON file"""
        with open(filename, 'r') as f:
            results = json.load(f)
        return results