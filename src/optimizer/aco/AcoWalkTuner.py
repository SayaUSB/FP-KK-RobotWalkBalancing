import logging
from ...placo.WalkParameters import WalkParameters
from ...evaluator.WalkFitnessEvaluator  import WalkFitnessEvaluator
from .AcoOptimization import ACOOptimizer
from typing import Dict, Optional, Callable

logger = logging.getLogger("aco_walk_tuner")


class ACOWalkTuner:
    """
    Main class untuk tuning walk parameters menggunakan ACO
    INTERFACE SAMA dengan PSOWalkTuner untuk consistency
    """
    
    def __init__(
        self,
        walk_params: WalkParameters,
        fitness_evaluator: WalkFitnessEvaluator,
        n_ants: int = 20,           # Analog n_particles
        max_iterations: int = 50,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q: float = 100.0,
    ):
        """
        Args:
            walk_params: WalkParameters object
            fitness_evaluator: WalkFitnessEvaluator object
            n_ants: Jumlah ants dalam colony
            max_iterations: Maximum iterasi optimization
            alpha: Pheromone importance
            beta: Heuristic importance
            rho: Evaporation rate
            q: Pheromone deposit factor
        """
        self.walk_params = walk_params
        self.fitness_evaluator = fitness_evaluator
        
        # Create ACO optimizer
        self.optimizer = ACOOptimizer(
            bounds=walk_params.get_tunable_bounds(),
            fitness_function=fitness_evaluator.evaluate,
            n_ants=n_ants,
            max_iterations=max_iterations,
            alpha=alpha,
            beta=beta,
            rho=rho,
            q=q,
        )
        
    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ):
        """Run ACO optimization (SAMA signature dengan PSO)"""
        logger.info("Starting ACO optimization for walk parameters...")
        best_params, best_fitness = self.optimizer.optimize(verbose=verbose, callback=callback)
        return best_params, best_fitness
        
    def apply_best_parameters(self):
        """Apply best parameters dari optimization (SAMA dengan PSO)"""
        best_params = self.optimizer.global_best_params
        if best_params:
            self.walk_params.set_parameters_from_dict(best_params)
            logger.info("Best parameters applied to walk_params")
        else:
            logger.warning("No optimization results available")
            
    def save_results(self, filename: str):
        """Save optimization results (SAMA dengan PSO)"""
        self.optimizer.save_results(filename)
        
    def get_optimization_history(self):
        """Get optimization history (SAMA dengan PSO)"""
        return self.optimizer.get_optimization_history()
