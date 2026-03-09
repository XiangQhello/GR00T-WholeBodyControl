#!/usr/bin/env python3
"""
Full pipeline: Convert CSV to SONIC-compatible pickle format

This script directly converts CSV motion data to SONIC format in one step.
It runs in Isaac Lab environment to compute forward kinematics for body parts.

Usage:
    python csv_to_sonic.py --input /path/to/motion.csv --output /path/to/output.pkl --name motion_name

Requirements:
    - Isaac Lab environment (source scripts/zeroth/zs_functional_shell.sh or similar)
    - The CSV format must match beyondmimic/LAFAN1 format:
      - Column 0: frame number (ignored)
      - Columns 1-3: root position (x, y, z)
      - Columns 4-7: root quaternion (x, y, z, w) -> converted to (w, x, y, z)
      - Columns 8+: 29 joint positions

Output format (SONIC pickle):
    - joint_pos: (timesteps, 29)
    - joint_vel: (timesteps, 29)
    - body_pos_w: (timesteps, 14, 3)
    - body_quat_w: (timesteps, 14, 4)
    - body_lin_vel_w: (timesteps, 14, 3)
    - body_ang_vel_w: (timesteps, 14, 3)
"""

import argparse
import os
import sys

# Isaac Lab imports
from isaaclab.app import AppLauncher
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
import torch
import numpy as np
import joblib


# Default URDF paths (will try multiple locations)
DEFAULT_URDF_PATHS = [
    "/home/jq/humanoid_robot/GR00T-WholeBodyControl/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf",
    "/home/jq/dance/whole_body_tracking/source/whole_body_tracking/assets/unitree_description/urdf/g1/g1_mode5.urdf",
    "/home/jq/humanoid_robot/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf",
    "./assets/unitree_description/urdf/g1/g1_mode5.urdf",
]

def find_urdf():
    """Find available G1 URDF file"""
    for path in DEFAULT_URDF_PATHS:
        if os.path.exists(path):
            print(f"  Found URDF: {path}")
            return path
    raise FileNotFoundError("Could not find G1 URDF file. Please specify with --urdf option")


# G1 Robot Configuration - will be set after finding URDF
G1_ROBOT_CFG = None

# 29 joints in G1 DOF order
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def create_scene_cfg(robot_cfg):
    """Create scene config with given robot config"""
    @configclass
    class MotionReplaySceneCfg(InteractiveSceneCfg):
        """Configuration for motion replay scene."""
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
        robot: ArticulationCfg = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    return MotionReplaySceneCfg


class CSVToSonicConverter:
    """Convert CSV motion data to SONIC format using Isaac Lab FK."""

    def __init__(self, csv_file, output_pkl, motion_name, input_fps=30, output_fps=50, frame_range=None):
        self.csv_file = csv_file
        self.output_pkl = output_pkl
        self.motion_name = motion_name
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.frame_range = frame_range
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps

    def load_csv(self):
        """Load and parse CSV motion data."""
        print(f"Loading CSV from: {self.csv_file}")

        # Load CSV (skip first column which is frame number)
        if self.frame_range is not None:
            data = np.loadtxt(
                self.csv_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
                usecols=range(1, 36)  # Skip first column (frame number), get 35 columns
            )
        else:
            data = np.loadtxt(self.csv_file, delimiter=",", usecols=range(1, 36))

        # Handle single row case
        if data.ndim == 1:
            data = data.reshape(1, -1)

        print(f"  Loaded {data.shape[0]} frames, {data.shape[1]} columns")

        # Parse: columns 0-2: root pos, 3-6: root rot (xyzw), 7+: joints
        root_pos = data[:, :3].astype(np.float32)
        root_rot_xyzw = data[:, 3:7].astype(np.float32)  # xyzw
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]  # Convert to wxyz
        joint_pos = data[:, 7:].astype(np.float32)

        print(f"  Root pos shape: {root_pos.shape}")
        print(f"  Root rot shape: {root_rot_wxyz.shape}")
        print(f"  Joint pos shape: {joint_pos.shape}")

        # Check joint count
        if joint_pos.shape[1] != 29:
            print(f"  WARNING: Expected 29 joints, got {joint_pos.shape[1]}")
            if joint_pos.shape[1] < 29:
                # Pad with zeros
                joint_pos = np.pad(joint_pos, ((0, 0), (0, 29 - joint_pos.shape[1])), mode='constant')
            else:
                # Truncate
                joint_pos = joint_pos[:, :29]

        # Interpolate to output FPS
        self.interpolate_motion(root_pos, root_rot_wxyz, joint_pos)

    def interpolate_motion(self, root_pos, root_rot, joint_pos):
        """Interpolate motion to target FPS."""
        input_frames = root_pos.shape[0]
        duration = (input_frames - 1) * self.input_dt
        output_frames = int(duration * self.output_fps) + 1

        print(f"  Interpolating: {input_frames} frames @ {self.input_fps}Hz -> {output_frames} frames @ {self.output_fps}Hz")

        # Create output time points
        times = np.arange(0, duration + 1e-6, self.output_dt)
        output_frames = len(times)

        # Compute interpolation indices
        phase = times / duration
        index_0 = (phase * (input_frames - 1)).astype(int)
        index_1 = np.minimum(index_0 + 1, input_frames - 1)
        blend = (phase * (input_frames - 1)) - index_0

        # Interpolate root position (linear)
        self.root_pos = root_pos[index_0] * (1 - blend).reshape(-1, 1) + root_pos[index_1] * blend.reshape(-1, 1)

        # Interpolate root rotation (SLERP)
        self.root_rot = np.zeros((output_frames, 4), dtype=np.float32)
        for i in range(output_frames):
            self.root_rot[i] = self._slerp(root_rot[index_0[i]], root_rot[index_1[i]], blend[i])

        # Interpolate joint position (linear)
        self.joint_pos = joint_pos[index_0] * (1 - blend).reshape(-1, 1) + joint_pos[index_1] * blend.reshape(-1, 1)

        print(f"  Output: {self.root_pos.shape[0]} frames")

    def _slerp(self, q1, q2, t):
        """Spherical linear interpolation."""
        q1 = torch.from_numpy(q1)
        q2 = torch.from_numpy(q2)
        result = quat_slerp(q1, q2, torch.tensor(t))
        return result.numpy()

    def compute_velocities(self):
        """Compute velocities from positions."""
        print("Computing velocities...")

        # Joint velocities (numerical differentiation)
        self.joint_vel = np.gradient(self.joint_pos, self.output_dt, axis=0)

        # Root linear velocity
        self.root_lin_vel = np.gradient(self.root_pos, self.output_dt, axis=0)

        # Root angular velocity (from quaternion derivative)
        self.root_ang_vel = self._compute_ang_vel_from_quat(self.root_rot)

        print(f"  joint_vel shape: {self.joint_vel.shape}")
        print(f"  root_lin_vel shape: {self.root_lin_vel.shape}")
        print(f"  root_ang_vel shape: {self.root_ang_vel.shape}")

    def _compute_ang_vel_from_quat(self, quats):
        """Compute angular velocity from quaternion sequence."""
        # Using finite difference of quaternions
        omega = np.zeros((len(quats), 3), dtype=np.float32)
        for i in range(len(quats) - 1):
            q1 = torch.from_numpy(quats[i])
            q2 = torch.from_numpy(quats[i + 1])
            q_rel = quat_mul(q2, quat_conjugate(q1)).numpy()
            omega[i + 1] = axis_angle_from_quat(torch.from_numpy(q_rel)).numpy() / self.output_dt

        return omega

    def run_fk_in_sim(self, sim, scene):
        """Run forward kinematics in simulation to get body part positions."""
        print("Running FK in simulation...")

        robot = scene["robot"]
        robot_joint_indexes = robot.find_joints(G1_JOINT_NAMES, preserve_order=True)[0]

        print(f"  Found {len(robot_joint_indexes)} joints in robot")
        print(f"  Joint indices: {robot_joint_indexes}")

        # Data storage
        num_frames = self.root_pos.shape[0]
        body_pos_w = []
        body_quat_w = []
        body_lin_vel_w = []
        body_ang_vel_w = []

        # Run simulation to get FK for each frame
        for frame_idx in range(num_frames):
            # Set root state
            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = torch.from_numpy(self.root_pos[frame_idx])
            root_states[:, 3:7] = torch.from_numpy(self.root_rot[frame_idx])
            root_states[:, 7:10] = torch.from_numpy(self.root_lin_vel[frame_idx])
            root_states[:, 10:] = torch.from_numpy(self.root_ang_vel[frame_idx])
            robot.write_root_state_to_sim(root_states)

            # Set joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos[:, robot_joint_indexes] = torch.from_numpy(self.joint_pos[frame_idx])
            joint_vel[:, robot_joint_indexes] = torch.from_numpy(self.joint_vel[frame_idx])
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # Update scene to compute FK
            scene.update(self.output_dt)

            # Record body data
            body_pos_w.append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            body_quat_w.append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            body_lin_vel_w.append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            body_ang_vel_w.append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

            if (frame_idx + 1) % 50 == 0:
                print(f"  Processed {frame_idx + 1}/{num_frames} frames")

        # Convert to numpy arrays
        body_pos_w = np.stack(body_pos_w, axis=0)
        body_quat_w = np.stack(body_quat_w, axis=0)
        body_lin_vel_w = np.stack(body_lin_vel_w, axis=0)
        body_ang_vel_w = np.stack(body_ang_vel_w, axis=0)

        print(f"  body_pos_w shape: {body_pos_w.shape}")
        print(f"  body_quat_w shape: {body_quat_w.shape}")
        print(f"  body_lin_vel_w shape: {body_lin_vel_w.shape}")
        print(f"  body_ang_vel_w shape: {body_ang_vel_w.shape}")

        # Get number of bodies
        num_bodies = body_pos_w.shape[1]
        print(f"  Number of bodies: {num_bodies}")

        return body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, num_bodies

    def save_output(self, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, num_bodies):
        """Save output in SONIC pickle format."""
        print(f"Saving to: {self.output_pkl}")

        # For SONIC, we need 14 body parts
        # If the robot has more, we'll keep the first 14
        if num_bodies > 14:
            print(f"  Truncating from {num_bodies} to 14 body parts")
            body_pos_w = body_pos_w[:, :14, :]
            body_quat_w = body_quat_w[:, :14, :]
            body_lin_vel_w = body_lin_vel_w[:, :14, :]
            body_ang_vel_w = body_ang_vel_w[:, :14, :]

        # Create output dictionary
        motion_data = {
            'joint_pos': self.joint_pos,
            'joint_vel': self.joint_vel,
            'body_pos_w': body_pos_w,
            'body_quat_w': body_quat_w,
            'body_lin_vel_w': body_lin_vel_w,
            'body_ang_vel_w': body_ang_vel_w,
            '_body_indexes': {f'body_{i}': i for i in range(min(num_bodies, 14))},
            'time_step_total': self.root_pos.shape[0],
        }

        output_data = {self.motion_name: motion_data}

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_pkl) if os.path.dirname(self.output_pkl) else '.', exist_ok=True)

        # Save as joblib pickle
        joblib.dump(output_data, self.output_pkl)

        print(f"  ✓ Successfully saved motion '{self.motion_name}'")
        print(f"    - Timesteps: {self.root_pos.shape[0]}")
        print(f"    - Joints: {self.joint_pos.shape[1]}")
        print(f"    - Body parts: {body_pos_w.shape[1]}")

    def convert(self, sim, scene):
        """Run full conversion pipeline."""
        # Step 1: Load and interpolate CSV
        self.load_csv()

        # Step 2: Compute velocities
        self.compute_velocities()

        # Step 3: Run FK in simulation
        body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, num_bodies = self.run_fk_in_sim(sim, scene)

        # Step 4: Save output
        self.save_output(body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, num_bodies)


def main():
    global G1_ROBOT_CFG

    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert CSV to SONIC format (FULL PIPELINE)')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output PKL file (will also generate CSV)')
    parser.add_argument('--name', '-n', type=str, default='motion', help='Motion name')
    parser.add_argument('--input-fps', type=int, default=30, help='Input FPS (default: 30)')
    parser.add_argument('--output-fps', type=int, default=50, help='Output FPS (default: 50)')
    parser.add_argument('--frame-range', nargs=2, type=int, default=None,
                        help='Frame range: START END (both inclusive, 1-indexed)')
    parser.add_argument('--urdf', '-u', type=str, default=None,
                        help='Path to G1 URDF file (auto-detect if not specified)')

    # Append Isaac Lab args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Find URDF file
    if args.urdf:
        urdf_path = args.urdf
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
    else:
        print("Searching for G1 URDF file...")
        urdf_path = find_urdf()

    # Create robot config
    G1_ROBOT_CFG = ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=urdf_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            joint_pos={},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
    )

    # Launch Isaac Lab
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(device=args.device, dt=1.0 / args.output_fps)
    sim = SimulationContext(sim_cfg)

    # Create scene
    scene_cfg = create_scene_cfg(G1_ROBOT_CFG)(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Reset simulation
    sim.reset()

    print("=" * 60)
    print("CSV to SONIC Converter")
    print("=" * 60)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Name:    {args.name}")
    print(f"FPS:     {args.input_fps} -> {args.output_fps}")
    if args.frame_range:
        print(f"Frames:  {args.frame_range[0]} - {args.frame_range[1]}")
    print("=" * 60)

    # Run conversion
    converter = CSVToSonicConverter(
        csv_file=args.input,
        output_pkl=args.output,
        motion_name=args.name,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        frame_range=args.frame_range
    )

    converter.convert(sim, scene)

    print("\n✓ PKL conversion complete!")
    print(f"\nNow generating CSV files...")

    # Generate CSV files automatically
    csv_output_dir = args.output.replace('.pkl', '')
    _convert_pkl_to_csv(args.output, csv_output_dir)

    print(f"\n{'='*60}")
    print(f"✓ FULL CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output directory: {csv_output_dir}/")
    print(f"")
    print(f"Files generated:")
    print(f"  - joint_pos.csv (29 joints)")
    print(f"  - joint_vel.csv")
    print(f"  - body_pos.csv (14 body parts)")
    print(f"  - body_quat.csv")
    print(f"  - body_lin_vel.csv")
    print(f"  - body_ang_vel.csv")
    print(f"")
    print(f"Run SONIC with:")
    motion_dir = os.path.basename(csv_output_dir)
    print(f"  cd gear_sonic_deploy")
    print(f"  echo 'y' | bash deploy.sh sim --motion-data reference/{motion_dir}/ --input-type keyboard")

    # Close simulation
    simulation_app.close()


def _convert_pkl_to_csv(pkl_file, output_dir):
    """Convert pkl to CSV files (same as convert_motions.py)"""
    # Add script directory to path for imports
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Import the conversion function from convert_motions.py
    from convert_motions import convert_motion_data

    print(f"Converting {pkl_file} to CSV files...")

    # Run conversion
    success, motion_count, joint_count, body_count = convert_motion_data(pkl_file, output_dir)

    if success:
        print(f"✓ Successfully converted to CSV format")
    else:
        print(f"✗ CSV conversion failed")


if __name__ == "__main__":
    main()
