import json
import psutil
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Callable, Optional, Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aco_optimizer")

class Ant:
    """Single ant in ACO colony"""
    
    def __init__(self, bounds: Dict[str, 'ParameterBounds']):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_dims = len(self.param_names)
        
        # Position (solution) - initialize randomly
        self.position = np.array([
            np.random.uniform(bounds[name].min_value, bounds[name].max_value)
            for name in self.param_names
        ])
        
        # Fitness
        self.fitness = float('inf')
        
    def to_dict(self) -> Dict[str, float]:
        """Convert position to parameter dictionary"""
        return {
            name: self.position[i]
            for i, name in enumerate(self.param_names)
        }
        
    def construct_solution(
        self,
        pheromone_matrix: np.ndarray,
        heuristic_matrix: np.ndarray,
        alpha: float,
        beta: float,
        n_discrete: int = 20
    ):
        """
        Construct solution menggunakan pheromone dan heuristic
        
        Args:
            pheromone_matrix: Matriks pheromone [n_dims, n_discrete]
            heuristic_matrix: Matriks heuristic [n_dims, n_discrete]
            alpha: Pheromone weight
            beta: Heuristic weight
            n_discrete: Number of discrete points per dimension
        """
        for i, name in enumerate(self.param_names):
            bounds = self.bounds[name]
            
            # Calculate probabilities ufor each discrete point
            tau = pheromone_matrix[i, :] ** alpha  # Pheromone
            eta = heuristic_matrix[i, :] ** beta   # Heuristic
            probabilities = tau * eta
            
            # Normalize
            prob_sum = probabilities.sum()
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                # Uniform if all are 0
                probabilities = np.ones(n_discrete) / n_discrete
            
            # Select discrete point based on probability
            selected_idx = np.random.choice(n_discrete, p=probabilities)
            
            # Convert discrete index into continuous value
            discrete_values = np.linspace(
                bounds.min_value,
                bounds.max_value,
                n_discrete
            )
            self.position[i] = discrete_values[selected_idx]
            
    def update_fitness(self, fitness: float):
        """Update fitness"""
        self.fitness = fitness


class ACOOptimizer:
    """
    Ant Colony Optimization algorithm for continuous optimization
    """
    
    def __init__(
        self,
        bounds: Dict[str, 'ParameterBounds'],
        fitness_function: Callable[[Dict[str, float]], float],
        n_ants: int = 20,           # Analog dengan n_particles
        max_iterations: int = 50,    # Max iterations
        alpha: float = 1.0,          # Pheromone weight
        beta: float = 2.0,           # Heuristic weight
        rho: float = 0.1,            # Pheromone evaporation rate
        q: float = 100.0,            # Pheromone deposit factor
        n_discrete: int = 20,        # Discrete points per dimension
    ):
        """
        Args:
            bounds: Dictionary of parameter bounds
            fitness_function: Function that receives dict parameters and return fitness score (lower is better)
            n_ants: Total ants in  colony (analog n_particles)
            max_iterations: Maximum iteration optimization
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Pheromone evaporation rate (0 to 1)
            q: Pheromone deposit factor
            n_discrete: Number of discrete points per dimension
        """
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.n_ants = n_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.n_discrete = n_discrete
        
        self.ants: List[Ant] = []
        self.n_dims = len(bounds)
        
        # Pheromone matrix [n_dims, n_discrete]
        self.pheromone_matrix = np.ones((self.n_dims, n_discrete))
        
        # Heuristic matrix (initialized uniform, updated during optimization)
        self.heuristic_matrix = np.ones((self.n_dims, n_discrete))
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.global_best_params = None
        
        # History
        self.fitness_history = []
        self.best_fitness_history = []
        self.iteration_best_params = []
        
    def initialize_colony(self):
        """Initialize ant colony (analog initialize_swarm di PSO)"""
        logger.info(f"Initializing colony with {self.n_ants} ants")
        self.ants = [Ant(self.bounds) for _ in range(self.n_ants)]
        
        # Initialize pheromone matrix
        self.pheromone_matrix = np.ones((self.n_dims, self.n_discrete))
        self.heuristic_matrix = np.ones((self.n_dims, self.n_discrete))
        
    def evaluate_ant(self, ant: Ant) -> float:
        """Evaluate fitness untuk ant (analog evaluate_particle)"""
        param_dict = ant.to_dict()
        fitness = self.fitness_function(param_dict)
        return fitness
        
    def update_pheromone(self):
        """Update pheromone matrix"""
        # Evaporation
        self.pheromone_matrix *= (1 - self.rho)
        
        # Deposit pheromone dari ants
        for ant in self.ants:
            if ant.fitness < float('inf'):
                # Pheromone amount inversely proportional to fitness (lower is better)
                pheromone_deposit = self.q / (1 + ant.fitness)
                
                # Add pheromone at discrete points that have been chosen by the ant
                for i, name in enumerate(ant.param_names):
                    bounds = self.bounds[name]
                    value = ant.position[i]
                    
                    # Find closest discrete point
                    discrete_values = np.linspace(
                        bounds.min_value,
                        bounds.max_value,
                        self.n_discrete
                    )
                    closest_idx = np.argmin(np.abs(discrete_values - value))
                    
                    # Deposit pheromone
                    self.pheromone_matrix[i, closest_idx] += pheromone_deposit
                    
        # Clamp pheromone values (prevent too high/low)
        self.pheromone_matrix = np.clip(
            self.pheromone_matrix,
            0.01,  # tau_min
            100.0  # tau_max
        )
        
    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ) -> Tuple[Dict[str, float], float]:
        """
        Run ACO optimization
        
        Args:
            verbose: Print progress
            callback: Function called after each iteration with (iteration, best_fitness, best_params)
            
        Returns:
            tuple: (best_parameters_dict, best_fitness)
        """
        self.initialize_colony()
        
        for iteration in range(self.max_iterations):
            iteration_fitnesses = []
            
            # Construct solutions for all ants
            for ant in self.ants:
                # Construct solution using pheromone
                ant.construct_solution(
                    self.pheromone_matrix,
                    self.heuristic_matrix,
                    self.alpha,
                    self.beta,
                    self.n_discrete
                )
                
                # Evaluate ant
                fitness = self.evaluate_ant(ant)
                ant.update_fitness(fitness)
                iteration_fitnesses.append(fitness)
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = ant.position.copy()
                    self.global_best_params = ant.to_dict()
                    
                    if verbose:
                        logger.info(
                            f"Iteration {iteration + 1}/{self.max_iterations}: "
                            f"New best fitness = {fitness:.6f}"
                        )
                        
            # Update pheromone matrix
            self.update_pheromone()
            
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
                
            if callback:
                callback(iteration, self.global_best_fitness, self.global_best_params)
                
        if verbose:
            logger.info(f"\nOptimization completed!")
            logger.info(f"Best fitness: {self.global_best_fitness:.6f}")
            logger.info(f"Best parameters:")
            for name, value in self.global_best_params.items():
                logger.info(f"  {name}: {value:.6f}")
                
        return self.global_best_params, self.global_best_fitness
        
    def get_optimization_history(self) -> Dict:
        """Get optimization history (SAMA format dengan PSO)"""
        return {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'iteration_best_params': self.iteration_best_params,
            'global_best_params': self.global_best_params,
            'global_best_fitness': self.global_best_fitness,
        }
        
    def save_results(self, filename: str):
        """Save optimization results to JSON file (SAMA dengan PSO)"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'ACO',
            'aco_config': {
                'n_ants': self.n_ants,
                'max_iterations': self.max_iterations,
                'alpha': self.alpha,
                'beta': self.beta,
                'rho': self.rho,
                'q': self.q,
                'n_discrete': self.n_discrete,
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
        """Load optimization results from JSON file (SAMA dengan PSO)"""
        with open(filename, 'r') as f:
            results = json.load(f)
        return results


# ============================================================================
# File: aco_optimizer_16gb.py
# Memory-safe ACO untuk RAM 16GB (analog pso_optimizer_16gb.py)
# ============================================================================

import gc
import psutil
import time

class MemorySafeACOOptimizer(ACOOptimizer):
    """ACO Optimizer dengan aggressive memory management (SAMA dengan PSO version)"""
    
    def __init__(
        self,
        bounds,
        fitness_function,
        n_ants=10,             # ← Reduced untuk 16GB
        max_iterations=30,     # ← Reduced
        alpha=1.0,
        beta=2.0,
        rho=0.1,
        q=100.0,
        n_discrete=20,
        save_interval=5,       # ← Save setiap 5 iterations
    ):
        super().__init__(
            bounds, fitness_function, n_ants, max_iterations,
            alpha, beta, rho, q, n_discrete
        )
        self.save_interval = save_interval
        
    def optimize(self, verbose=True, callback=None):
        """Optimized version dengan periodic cleanup (SAMA struktur dengan PSO)"""
        self.initialize_colony()
        
        for iteration in range(self.max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration + 1}/{self.max_iterations}")
            logger.info(f"{'='*60}")
            
            # ✅ Memory check (SAMA dengan PSO)
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            vm = psutil.virtual_memory()
            
            logger.info(
                f"Memory: {mem_before:.0f}MB used, "
                f"{vm.available/1024/1024:.0f}MB available "
                f"({vm.percent:.1f}% system usage)"
            )
            
            # ✅ Check RAM kritis
            if vm.available < 2 * 1024 * 1024 * 1024:  # < 2GB
                logger.error("⚠️ RAM CRITICAL! Forcing cleanup...")
                gc.collect()
                time.sleep(0.5)
            
            iteration_fitnesses = []
            
            # Construct and evaluate ants
            for i, ant in enumerate(self.ants):
                logger.info(f"  Evaluating ant {i+1}/{self.n_ants}...")
                
                try:
                    # Construct solution
                    ant.construct_solution(
                        self.pheromone_matrix,
                        self.heuristic_matrix,
                        self.alpha,
                        self.beta,
                        self.n_discrete
                    )
                    
                    # Evaluate
                    fitness = self.evaluate_ant(ant)
                    ant.update_fitness(fitness)
                    iteration_fitnesses.append(fitness)
                    
                    # Update global best
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = ant.position.copy()
                        self.global_best_params = ant.to_dict()
                        
                        logger.info(f"  ★ NEW BEST: fitness = {fitness:.4f}")
                        
                except Exception as e:
                    logger.error(f"  ✗ Ant {i+1} failed: {e}")
                    iteration_fitnesses.append(10000.0)
                    
            # Update pheromone
            self.update_pheromone()
            
            # Record history
            avg_fitness = np.mean(iteration_fitnesses)
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(self.global_best_fitness)
            self.iteration_best_params.append(self.global_best_params.copy())
            
            logger.info(
                f"\n  Avg fitness: {avg_fitness:.4f} | "
                f"Best fitness: {self.global_best_fitness:.4f}"
            )
            
            # ✅ Periodic save
            if (iteration + 1) % self.save_interval == 0:
                self.save_results(f'aco_checkpoint_iter{iteration+1}.json')
                logger.info(f"  ✓ Checkpoint saved")
            
            # Callback
            if callback:
                callback(iteration, self.global_best_fitness, self.global_best_params)
                
            # ✅ Cleanup
            gc.collect()
            
        logger.info(f"\n{'='*60}")
        logger.info("OPTIMIZATION COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Best fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best_params, self.global_best_fitness
