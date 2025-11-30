import random
import math
import numpy as np
import placo

class PSOOptimizer:
    """
    Particle Swarm Optimization (PSO) for humanoid walking parameters.
    Each particle represents a candidate parameter vector normalized in [0,1].
    Fitness function: distance forward in PyBullet simulation penalized by falls.
    """

    def __init__(self, sim, initial_setup, dim=4, swarm_size=12, iters=18, dt=0.005):
        """
        sim: Simulation instance (PyBullet)
        initial_setup: dict containing robot, solver, tasks, WalkPatternGeneratorClass, parameters
        dim: number of parameters to optimize
        swarm_size: number of particles
        iters: number of iterations
        dt: simulation timestep
        """
        self.sim = sim
        self.initial_setup = initial_setup
        self.dim = dim
        self.swarm_size = swarm_size
        self.iters = iters
        self.dt = dt

        self.swarm = []
        for _ in range(swarm_size):
            pos = [random.random() for _ in range(dim)]
            vel = [random.uniform(-0.1, 0.1) for _ in range(dim)]
            self.swarm.append({
                "pos": pos,
                "vel": vel,
                "best_pos": pos[:],
                "best_val": -1e9
            })

        self.gbest_pos = None
        self.gbest_val = -1e9

        # PSO hyperparameters
        self.w = 0.7    # inertia
        self.c1 = 1.4   # cognitive
        self.c2 = 1.4   # social

        # Fall detection threshold (CoM z)
        self.FALL_COM_Z_THRESHOLD = 0.3

    def get_sim_robot_id(self):
        sim = self.sim
        # coba beberapa nama atribut umum
        for attr in ("robot_id", "robotId", "robot"):
            if hasattr(sim, attr):
                return getattr(sim, attr)
        # fallback: scan body names di PyBullet
        import pybullet as p
        for i in range(p.getNumBodies()):
            name = p.getBodyInfo(i)[1].decode('utf-8')
            if "robot" in name.lower() or "sigmaban" in name.lower():
                return i
        raise RuntimeError("Cannot find robot id in Simulation")

    def evaluate_params(self, vec):
        """
        Map normalized vector to real parameters, run short simulation, and return fitness.
        vec: list of values in [0,1]^dim
        """
        robot = self.initial_setup["robot"]
        solver = self.initial_setup["solver"]
        tasks = self.initial_setup["tasks"]
        WalkPatternGeneratorClass = self.initial_setup["WalkPatternGeneratorClass"]
        parameters = self.initial_setup["parameters"]
        DT = self.dt

        # Map normalized vector to real parameter ranges
        ss_dur = 0.15 + vec[0] * (0.6 - 0.15)            # single_support_duration
        com_h   = 0.25 + vec[1] * (0.40 - 0.25)          # walk_com_height
        trunk_p = 0.0  + vec[2] * (0.35 - 0.0)           # walk_trunk_pitch
        max_dx  = 0.05 + vec[3] * (0.5 - 0.05)           # walk_max_dx_forward

        # Apply parameters
        parameters.single_support_duration = ss_dur
        parameters.single_support_timesteps = max(3, int(round(ss_dur / DT)))
        parameters.walk_com_height = com_h
        parameters.walk_trunk_pitch = trunk_p
        parameters.walk_max_dx_forward = max_dx

        # Footstep planner (short for evaluation)
        repetitive = placo.FootstepsPlannerRepetitive(parameters)
        d_x = min(max_dx * 0.8, max_dx)
        repetitive.configure(d_x, 0.0, 0.0, 6)
        T_world_left = placo.flatten_on_floor(robot.get_T_world_left())
        T_world_right = placo.flatten_on_floor(robot.get_T_world_right())
        footsteps = repetitive.plan(placo.HumanoidRobot_Side.left, T_world_left, T_world_right)
        supports = placo.FootstepsPlanner.make_supports(
            footsteps, 0.0, True, parameters.has_double_support(), True
        )

        # Walk planner & trajectory
        walk = WalkPatternGeneratorClass(robot, parameters)
        try:
            trajectory = walk.plan(supports, robot.com_world(), 0.0)
        except Exception:
            return -1e6  # gagal perencanaan → fitness buruk

        # Sim robot ID
        robot_id = self.get_sim_robot_id()

        import pybullet as p
        # Initial CoM x for distance
        init_pos = p.getBasePositionAndOrientation(robot_id)[0]

        fallen = False
        fall_penalty = 0.0
        eval_time = min(6.0, ss_dur * 6 * 1.2)
        steps = int(math.ceil(eval_time / DT))

        for i in range(steps):
            tasks.update_tasks_from_trajectory(trajectory, i * DT)
            robot.update_kinematics()
            qd_sol = solver.solve(True)

            try:
                joints = {joint: robot.get_joint(joint) for joint in self.sim.getJoints()}
                self.sim.setJoints(joints)
            except Exception:
                pass

            self.sim.tick()
            com_pos = p.getBasePositionAndOrientation(robot_id)[0]
            com_z = com_pos[2]
            if com_z < self.FALL_COM_Z_THRESHOLD:
                fallen = True
                fall_penalty = 5.0
                break

        final_pos = p.getBasePositionAndOrientation(robot_id)[0]
        dist_x = final_pos[0] - init_pos[0]

        fitness = dist_x - fall_penalty
        return fitness

    def run(self):
        for it in range(self.iters):
            print(f"[PSO] Iter {it+1}/{self.iters}")
            for particle in self.swarm:
                val = self.evaluate_params(particle["pos"])
                if val is None:
                    val = -1e9
                # personal best
                if val > particle["best_val"]:
                    particle["best_val"] = val
                    particle["best_pos"] = particle["pos"][:]
                # global best
                if val > self.gbest_val:
                    self.gbest_val = val
                    self.gbest_pos = particle["pos"][:]
                print(f"  -> val={val:.4f}")

            # update velocity & position
            for particle in self.swarm:
                for d in range(self.dim):
                    r1 = random.random()
                    r2 = random.random()
                    cognitive = self.c1 * r1 * (particle["best_pos"][d] - particle["pos"][d])
                    social    = self.c2 * r2 * (self.gbest_pos[d] - particle["pos"][d]) if self.gbest_pos else 0.0
                    particle["vel"][d] = self.w * particle["vel"][d] + cognitive + social
                    particle["vel"][d] = max(min(particle["vel"][d], 0.5), -0.5)
                    particle["pos"][d] += particle["vel"][d]
                    particle["pos"][d] = max(min(particle["pos"][d], 1.0), 0.0)

        # Map best normalized vector to real parameter values
        best_vec = self.gbest_pos
        best_params = {
            "single_support_duration": 0.15 + best_vec[0] * (0.6 - 0.15),
            "walk_com_height": 0.25 + best_vec[1] * (0.40 - 0.25),
            "walk_trunk_pitch": 0.0  + best_vec[2] * (0.35 - 0.0),
            "walk_max_dx_forward": 0.05 + best_vec[3] * (0.5 - 0.05)
        }
        print("[PSO] Done. Best fitness:", self.gbest_val)
        print("[PSO] Best params:", best_params)
        return best_params
