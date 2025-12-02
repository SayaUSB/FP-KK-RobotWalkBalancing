import numpy as np
import logging
from typing import Dict, Callable, Optional, Tuple, List
import json
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ga_optimizer")

class Individual:
    """Single individual (chromosome) in the GA population"""
    
    def __init__(self, bounds: Dict[str, 'ParameterBounds']):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        
        # Initialize genes randomly within bounds
        self.genes = {}
        for name in self.param_names:
            b = bounds[name]
            self.genes[name] = np.random.uniform(b.min_value, b.max_value)
            
        self.fitness = float('inf')
        
    def to_dict(self) -> Dict[str, float]:
        """Convert genes to parameter dictionary"""
        return self.genes.copy()
    
    def from_dict(self, params: Dict[str, float]):
        """Load genes from dictionary"""
        self.genes = params.copy()

class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for continuous parameter optimization.
    Uses Tournament Selection, Uniform Crossover, and Gaussian Mutation.
    """
    
    def __init__(
        self,
        bounds: Dict[str, 'ParameterBounds'],
        fitness_function: Callable[[Dict[str, float]], float],
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
        mutation_scale: float = 0.1  # Scale of gaussian noise relative to bounds range
    ):
        """
        Args:
            bounds: Dictionary of parameter bounds
            fitness_function: Function returning fitness (lower is better)
            population_size: Number of individuals
            max_generations: Max iterations
            mutation_rate: Probability of a gene mutating
            crossover_rate: Probability of parents crossing over
            elite_size: Number of best individuals to carry over unchanged
            mutation_scale: Strength of mutation (0.1 = 10% of parameter range)
        """
        self.bounds = bounds
        self.fitness_function = fitness_function
        self.pop_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.mutation_scale = mutation_scale
        
        self.population: List[Individual] = []
        self.param_names = list(bounds.keys())
        
        # Global best tracking
        self.global_best_fitness = float('inf')
        self.global_best_params = None
        
        # History
        self.fitness_history = []
        self.best_fitness_history = []
        
    def initialize_population(self):
        """Initialize random population"""
        logger.info(f"Initializing population with {self.pop_size} individuals")
        self.population = [Individual(self.bounds) for _ in range(self.pop_size)]
        
    def evaluate_population(self):
        """Evaluate fitness for all individuals"""
        for ind in self.population:
            # Skip if already evaluated (optimization for elites)
            if ind.fitness != float('inf'):
                continue
                
            fitness = self.fitness_function(ind.to_dict())
            ind.fitness = fitness
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_params = ind.to_dict()

    def tournament_selection(self, k=3) -> Individual:
        """Select best individual from k random individuals"""
        candidates = random.sample(self.population, k)
        # Return the one with lowest fitness (minimization)
        return min(candidates, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform Crossover: Child inherits genes randomly from P1 or P2"""
        child = Individual(self.bounds)
        
        if np.random.random() < self.crossover_rate:
            for name in self.param_names:
                # 50% chance from P1, 50% from P2
                if np.random.random() < 0.5:
                    child.genes[name] = parent1.genes[name]
                else:
                    child.genes[name] = parent2.genes[name]
        else:
            # If no crossover, clone parent1
            child.genes = parent1.genes.copy()
            
        return child

    def mutate(self, individual: Individual):
        """Gaussian Mutation: Add small noise to genes"""
        for name in self.param_names:
            if np.random.random() < self.mutation_rate:
                b = self.bounds[name]
                param_range = b.max_value - b.min_value
                
                # Add Gaussian noise
                noise = np.random.normal(0, param_range * self.mutation_scale)
                new_val = individual.genes[name] + noise
                
                # Clip to bounds
                individual.genes[name] = np.clip(new_val, b.min_value, b.max_value)

    def optimize(
        self,
        verbose: bool = True,
        callback: Optional[Callable[[int, float, Dict[str, float]], None]] = None
    ) -> Tuple[Dict[str, float], float]:
        """Run Genetic Algorithm"""
        self.initialize_population()
        self.evaluate_population()
        
        for generation in range(self.max_generations):
            # 1. Sort population by fitness (ascending/lowest is best)
            self.population.sort(key=lambda x: x.fitness)
            
            current_best = self.population[0].fitness
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            
            # Record history
            self.fitness_history.append(avg_fitness)
            self.best_fitness_history.append(self.global_best_fitness)
            
            if verbose:
                logger.info(
                    f"Gen {generation + 1}/{self.max_generations}: "
                    f"Avg = {avg_fitness:.4f}, "
                    f"Best = {self.global_best_fitness:.4f}"
                )
            
            if callback:
                callback(generation, self.global_best_fitness, self.global_best_params)

            # 2. Elitism: Keep best individuals
            new_population = []
            for i in range(self.elite_size):
                # Deep copy to ensure they aren't mutated later
                elite = Individual(self.bounds)
                elite.genes = self.population[i].genes.copy()
                elite.fitness = self.population[i].fitness
                new_population.append(elite)
                
            # 3. Generate new population
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                self.mutate(child)
                
                new_population.append(child)
                
            self.population = new_population
            
            # Evaluate new population
            self.evaluate_population()
            
        if verbose:
            logger.info(f"\nGA Optimization completed!")
            logger.info(f"Best fitness: {self.global_best_fitness:.6f}")
            
        return self.global_best_params, self.global_best_fitness

    def save_results(self, filename: str):
        """Save results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'Genetic Algorithm',
            'config': {
                'pop_size': self.pop_size,
                'generations': self.max_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size
            },
            'best_parameters': self.global_best_params,
            'best_fitness': float(self.global_best_fitness),
            'history': {
                'avg_fitness': [float(f) for f in self.fitness_history],
                'best_fitness': [float(f) for f in self.best_fitness_history]
            }
        }
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")

    def get_optimization_history(self) -> Dict:
        return {
            'fitness_history': self.fitness_history,
            'best_fitness_history': self.best_fitness_history,
            'global_best_params': self.global_best_params,
            'global_best_fitness': self.global_best_fitness,
        }