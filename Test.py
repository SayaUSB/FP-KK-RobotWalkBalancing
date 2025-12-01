import sys
import time
import json
import pathlib
import argparse
from src.placo.HumanoidWalkController import HumanoidWalkController
from src.placo.SimulationManager import SimulationManager

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Humanoid walking controller")
    parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
    parser.add_argument("-m", "--meshcat", action="store_true", help="MeshCat visualization")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to JSON config file")
    args = parser.parse_args()
    
    if not args.pybullet and not args.meshcat:
        print("No visualization selected, use either -p or -m")
        return
    if not args.config:
        parameters = None
    else:
        pcfg = None
        p = pathlib.Path(args.config)
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception as e:
                print(f"Failed to read JSON file '{p}': {e}")
                sys.exit(1)
        else:
            try:
                cfg = json.loads(args.config)
            except json.JSONDecodeError:
                print(f"Config path tidak ditemukan dan string bukan JSON valid: {args.config!r}")
                sys.exit(1)

        parameters = cfg.get("best_parameters")
        if parameters is None:
            print("JSON config tidak berisi key 'best_parameters'.")
            sys.exit(1)
        
    # Initialize controller
    model_filename = "model/sigmaban/robot.urdf"
    controller = HumanoidWalkController(model_filename, parameters=parameters)
    
    # Setup robot
    controller.initialize_robot_pose()
    controller.plan_initial_trajectory(d_x=0.1, d_y=0.0, d_theta=0.1, nb_steps=10)
    
    # Setup simulation/visualization
    sim_manager = SimulationManager(
        controller,
        use_pybullet=args.pybullet,
        use_meshcat=args.meshcat
    )
    
    # Main control loop
    while True:
        # Get CoM position
        com_z = sim_manager.get_com_position()
        
        # Check stability
        if not controller.check_stability(com_z):
            time.sleep(1)
            break
            
        # Execute one control step
        controller.step()
        
        # Update visualization
        supports = controller.replan_if_needed()
        sim_manager.update_pybullet(controller.t)
        sim_manager.update_meshcat(supports)
        
        # Timing control
        sim_manager.spin_lock(controller.t)


if __name__ == "__main__":
    main()