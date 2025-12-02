import os
import time
import tracemalloc
import threading
from src.placo.WalkParameters import WalkParameters
from src.evaluator.WalkFitnessEvaluator import WalkFitnessEvaluator
from src.optimizer.pso.ParticleSwarmWalkTuner import PSOWalkTuner
import matplotlib.pyplot as plt

def get_next_run_directory(base_dir="log/pso"):
    """
    Mendeteksi folder log terakhir dan membuat folder baru dengan nomor urut berikutnya.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        next_id = 1
    else:
        existing_dirs = [d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
        if existing_dirs:
            last_id = max([int(d) for d in existing_dirs])
            next_id = last_id + 1
        else:
            next_id = 1
    
    run_dir = os.path.join(base_dir, str(next_id))
    os.makedirs(run_dir)
    print(f"Created new log directory: {run_dir}")
    return run_dir

def monitor_memory_usage(memory_history, stop_event, start_time):
    while not stop_event.is_set():
        current, _ = tracemalloc.get_traced_memory()
        elapsed_time = time.time() - start_time
        memory_history.append((elapsed_time, current / (1024 * 1024)))
        time.sleep(0.1)  # Sampling rate: 100ms

def plot_optimization_results(history, memory_history, save_path):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    # Fitness Convergence
    ax1.plot(history['fitness_history'], label='Average Fitness', alpha=0.7)
    ax1.plot(history['best_fitness_history'], label='Best Fitness', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Fitness')
    ax1.set_title('PSO Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Fitness Improvement
    best_fitness = history['best_fitness_history']
    improvement = [0] + [best_fitness[i-1] - best_fitness[i] for i in range(1, len(best_fitness))]
    ax2.bar(range(len(improvement)), improvement, alpha=0.7, color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fitness Improvement')
    ax2.set_title('Improvement per Iteration')
    ax2.grid(True, alpha=0.3)
    
    # Memory Usage over Time 
    if memory_history:
        times = [m[0] for m in memory_history]
        mems = [m[1] for m in memory_history]
        
        ax3.plot(times, mems, label='Memory Usage (MB)', color='purple')
        ax3.fill_between(times, mems, color='purple', alpha=0.1)
        
        # Calculate Stats
        peak_mem = max(mems)
        avg_mem = sum(mems) / len(mems)
        
        # Add Reference Lines
        ax3.axhline(y=peak_mem, color='red', linestyle='--', alpha=0.7, label=f'Peak: {peak_mem:.2f} MB')
        ax3.axhline(y=avg_mem, color='green', linestyle='-.', alpha=0.7, label=f'Avg: {avg_mem:.2f} MB')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Memory (MB)')
        ax3.set_title('System Memory Usage during Optimization')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(save_path, 'pso_analysis_report.png')
    plt.savefig(plot_file)
    print(f"Analysis plot saved to {plot_file}")
    plt.close()

def format_time(seconds):
    """Mengubah detik menjadi format Jam:Menit:Detik"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.2f}s"

def main():
    """Main example"""
    from src.placo.HumanoidWalkController import HumanoidWalkController
    
    # Setup Logging & Profiling
    current_log_dir = get_next_run_directory("log/pso")
    
    # Mulai tracking memori
    print("Starting resource monitoring...")
    tracemalloc.start()
    start_time = time.time()
    
    # Setup Thread Monitoring Memory
    memory_history = []
    stop_monitoring = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_memory_usage, 
        args=(memory_history, stop_monitoring, start_time)
    )
    monitor_thread.start()

    try:
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
    
    finally:
        # STOP PROFILING
        stop_monitoring.set()
        monitor_thread.join()
        
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    total_time = end_time - start_time
    peak_mem_mb = peak_mem / (1024 * 1024)
    
    # Calculate average from history if available
    if memory_history:
        avg_mem_mb = sum(m[1] for m in memory_history) / len(memory_history)
    else:
        avg_mem_mb = current_mem / (1024 * 1024)
    
    # 5. Save results
    print(f"\nSaving results to {current_log_dir}...")
    tuner.save_results(os.path.join(current_log_dir, 'pso_results.json'))
    walk_params.save_to_file(os.path.join(current_log_dir, 'best_walk_params.json'))
    
    # 6. Plot results
    print("\nPlotting results...")
    history = tuner.get_optimization_history()
    plot_optimization_results(history, memory_history, current_log_dir)
    
    # 7. Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    
    # Performance Stats
    print(f"PERFORMANCE METRICS:")
    print(f"  Total Duration     : {format_time(total_time)}")
    print(f"  Peak Memory Usage  : {peak_mem_mb:.2f} MB")
    print(f"  Average Memory Usage: {avg_mem_mb:.2f} MB")
    print("-" * 60)
    
    print(f"Log Directory: {current_log_dir}")
    print(f"Best Fitness : {best_fitness:.6f}")
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