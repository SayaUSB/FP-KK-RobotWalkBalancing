from src.placo.WalkParameters import WalkParameters
from src.evaluator.WalkFitnessEvaluator import WalkFitnessEvaluator
from src.optimizer.pso.ParticleSwarmWalkTuner import PSOWalkTuner
import matplotlib.pyplot as plt


def plot_optimization_results(history):
    """Plot convergence history"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot average and best fitness
    ax1.plot(history['fitness_history'], label='Average Fitness', alpha=0.7)
    ax1.plot(history['best_fitness_history'], label='Best Fitness', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness')
    ax1.set_title('PSO Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot improvement rate
    best_fitness = history['best_fitness_history']
    improvement = [0] + [best_fitness[i-1] - best_fitness[i] for i in range(1, len(best_fitness))]
    ax2.bar(range(len(improvement)), improvement, alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness Improvement')
    ax2.set_title('Improvement per Iteration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pso_convergence.png')
    print("Convergence plot saved to pso_convergence.png")


def main():
    """Main example"""
    from src.placo.HumanoidWalkController import HumanoidWalkController
    
    # 1. Create walk parameters
    print("Creating walk parameters...")
    walk_params = WalkParameters()
    
    # 2. Create fitness evaluator
    print("Creating fitness evaluator...")
    evaluator = WalkFitnessEvaluator(
        controller_class=HumanoidWalkController,
        model_filename="model/sigmaban/robot.urdf",
        simulation_duration=5.0,
        target_distance=0.5,
        weights={
            'fall_penalty': 1000.0,
            'distance_error': 100.0,
            'instability': 50.0,
            'parameter_smoothness': 10.0,
        }
    )
    
    # 3. Create PSO tuner
    print("Creating PSO tuner...")
    tuner = PSOWalkTuner(
        walk_params=walk_params,
        fitness_evaluator=evaluator,
        n_particles=20,
        max_iterations=50,
        w=0.7,
        c1=1.5,
        c2=1.5,
    )
    
    # 4. Run optimization
    print("\nStarting optimization...")
    best_params, best_fitness = tuner.optimize(verbose=True)
    
    # 5. Save results
    print("\nSaving results...")
    tuner.save_results('pso_results.json')
    walk_params.save_to_file('best_walk_params.json')
    
    # 6. Plot results
    print("\nPlotting results...")
    history = tuner.get_optimization_history()
    plot_optimization_results(history)
    
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
