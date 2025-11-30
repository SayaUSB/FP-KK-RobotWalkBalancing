import placo

class FootstepPlanner:
    """Perencanaan footstep untuk walking"""
    
    def __init__(self, parameters):
        self.parameters = parameters
        self.planner = placo.FootstepsPlannerRepetitive(parameters)
        
    def configure_walk(self, d_x=0.1, d_y=0.0, d_theta=0.0, nb_steps=10):
        """
        Konfigurasi parameter berjalan
        
        Args:
            d_x: Jarak langkah forward/backward (m)
            d_y: Jarak langkah lateral (m)
            d_theta: Rotasi per langkah (rad)
            nb_steps: Jumlah langkah
        """
        self.planner.configure(d_x, d_y, d_theta, nb_steps)
        
    def plan_footsteps(self, robot, start_side=placo.HumanoidRobot_Side.left):
        """
        Plan footsteps dari posisi robot saat ini
        
        Args:
            robot: Object HumanoidRobot
            start_side: Sisi kaki yang memulai langkah
            
        Returns:
            tuple: (footsteps, supports)
        """
        T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
        T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
        
        footsteps = self.planner.plan(start_side, T_world_left, T_world_right)
        supports = placo.FootstepsPlanner.make_supports(
            footsteps, 0.0, True, 
            self.parameters.has_double_support(), True
        )
        
        return footsteps, supports