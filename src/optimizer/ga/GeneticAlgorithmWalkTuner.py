import logging
from typing import Optional, Callable, Dict
from ...placo.WalkParameters import WalkParameters
from .GeneticAlgorithmOptimization import GeneticAlgorithmOptimizer
from ...evaluator.WalkFitnessEvaluator import WalkFitnessEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ga_walk_tuner")

class GAWalkTuner:
    """
    Main class for tuning walk parameters using Genetic Algorithm.
    Follows the same interface as PSOWalkTuner.
    """
    
    def __init__(
        self,
        walk_params: WalkParameters,
        fitness_evaluator: WalkFitnessEvaluator,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
    ):
        """
        Args:
            walk_params: WalkParameters object
            fitness_evaluator: WalkFitnessEvaluator object
            population_size: Number of individuals in population
            max_generations: Maximum generations to run
            mutation_rate: Probability of gene mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top individuals to preserve
        """
        self.walk_params = walk_params
        self.fitness_evaluator = fitness_evaluator
        
        # Create GA optimizer
        self.optimizer = GeneticAlgorithmOptimizer(
            bounds=walk_params.get_tunable_bounds(),
            fitness_function=fitness_evaluator.evaluate,
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_size=elite_size
        )
        
    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ):
        """Run GA optimization"""
        logger.info("Starting Genetic Algorithm optimization for walk parameters...")
        best_params, best_fitness = self.optimizer.optimize(verbose=verbose, callback=callback)
        return best_params, best_fitness
        
    def apply_best_parameters(self):
        """Apply best parameters from optimization to walk_params"""
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