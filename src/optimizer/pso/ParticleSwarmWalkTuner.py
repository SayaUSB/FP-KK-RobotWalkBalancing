import logging
from typing import Optional, Callable, Dict
from ...placo.WalkParameters import WalkParameters
from .ParticleSwarmOptimization import PSOOptimizer
from ...evaluator.WalkFitnessEvaluator import WalkFitnessEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pso_walk_tuner")


class PSOWalkTuner:
    """Main class untuk tuning walk parameters menggunakan PSO"""
    
    def __init__(
        self,
        walk_params: WalkParameters,
        fitness_evaluator: WalkFitnessEvaluator,
        n_particles: int = 20,
        max_iterations: int = 50,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        """
        Args:
            walk_params: WalkParameters object
            fitness_evaluator: WalkFitnessEvaluator object
            n_particles: Jumlah particles dalam swarm
            max_iterations: Maximum iterasi optimization
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        self.walk_params = walk_params
        self.fitness_evaluator = fitness_evaluator
        
        # Create PSO optimizer
        self.optimizer = PSOOptimizer(
            bounds=walk_params.get_tunable_bounds(),
            fitness_function=fitness_evaluator.evaluate,
            n_particles=n_particles,
            max_iterations=max_iterations,
            w=w,
            c1=c1,
            c2=c2,
        )
        
    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ):
        """Run PSO optimization"""
        logger.info("Starting PSO optimization for walk parameters...")
        best_params, best_fitness = self.optimizer.optimize(verbose=verbose, callback=callback)
        return best_params, best_fitness
        
    def apply_best_parameters(self):
        """Apply best parameters dari optimization ke walk_params"""
        best_params = self.optimizer.global_best_params
        if best_params:
            self.walk_params.set_parameters_from_dict(best_params)
            logger.info("Best parameters applied to walk_params")
        else:
            logger.warning("No optimization results available")
            
    def save_results(self, filename: str):
        """Save optimization results"""
        self.optimizer.save_results(filename)
        
    def get_optimization_history(self):
        """Get optimization history"""
        return self.optimizer.get_optimization_history()