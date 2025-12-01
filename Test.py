import time
import argparse
from src.placo.HumanoidWalkController import HumanoidWalkController
from src.placo.SimulationManager import SimulationManager

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Humanoid walking controller")
    parser.add_argument("-p", "--pybullet", action="store_true", help="PyBullet simulation")
    parser.add_argument("-m", "--meshcat", action="store_true", help="MeshCat visualization")
    args = parser.parse_args()
    
    if not args.pybullet and not args.meshcat:
        print("No visualization selected, use either -p or -m")
        return
        
    # Initialize controller
    model_filename = "model/sigmaban/robot.urdf"
    controller = HumanoidWalkController(model_filename)
    
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