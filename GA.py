import os
import matplotlib.pyplot as plt
from src.evaluator.WalkFitnessEvaluator import WalkFitnessEvaluator
from src.placo.HumanoidWalkController import HumanoidWalkController
from src.placo.WalkParameters import WalkParameters
from src.optimizer.ga.GeneticAlgorithmWalkTuner import GAWalkTuner

def ensure_log_dir(directory):
    """Ensure logging directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_optimization_results(history, save_path):
    """Plot GA convergence history"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot average and best fitness
    ax1.plot(history['fitness_history'], label='Average Fitness', alpha=0.7, color='blue')
    ax1.plot(history['best_fitness_history'], label='Best Fitness', linewidth=2, color='red')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Genetic Algorithm Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot improvement rate
    best_fitness = history['best_fitness_history']
    improvement = [0] + [best_fitness[i-1] - best_fitness[i] for i in range(1, len(best_fitness))]
    ax2.bar(range(len(improvement)), improvement, alpha=0.7, color='green')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitness Improvement')
    ax2.set_title('Improvement per Generation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")

def main():
    # Setup log directory
    log_dir = 'log/ga'
    ensure_log_dir(log_dir)

    # 1. Create walk parameters
    print("Creating walk parameters...")
    walk_params = WalkParameters()
    
    # 2. Create fitness evaluator
    print("Creating fitness evaluator...")
    evaluator = WalkFitnessEvaluator(
        controller_class=HumanoidWalkController,
        model_filename="model/sigmaban/robot.urdf",
        simulation_duration=30,
        target_distance=5,
        weights={
            'fall_penalty': 1000.0,
            'distance_error': 100.0,
            'instability': 50.0,
            'parameter_smoothness': 10.0,
        }
    )
    
    # 3. Create GA tuner
    print("Creating GA tuner...")
    tuner = GAWalkTuner(
        walk_params=walk_params,
        fitness_evaluator=evaluator,
        population_size=20,     # Size of the population
        max_generations=50,     # Max iterations
        mutation_rate=0.1,      # 10% chance for a gene to mutate
        crossover_rate=0.8,     # 80% chance for crossover
        elite_size=2            # Keep top 2 best individuals
    )
    
    # 4. Run optimization
    print("\nStarting optimization...")
    best_params, best_fitness = tuner.optimize(verbose=True)
    
    # 5. Save results
    print("\nSaving results...")
    results_file = os.path.join(log_dir, 'ga_results.json')
    params_file = os.path.join(log_dir, 'best_walk_params_ga.json')
    plot_file = os.path.join(log_dir, 'ga_convergence.png')
    
    tuner.save_results(results_file)
    walk_params.save_to_file(params_file)
    
    # 6. Plot results
    print("\nPlotting results...")
    history = tuner.get_optimization_history()
    plot_optimization_results(history, plot_file)
    
    # 7. Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Best Fitness: {best_fitness:.6f}")
    print("\nBest Parameters:")
    for name, value in best_params.items():
        bounds = walk_params.tunable_params[name]
        print(f"  {name:30s}: {value:.6f} (default: {bounds.default_value:.6f})")
    print("="*60)
    
    # 8. Apply best parameters
    tuner.apply_best_parameters()
    print("\nBest parameters applied to walk_params object")

if __name__ == "__main__":
    main()