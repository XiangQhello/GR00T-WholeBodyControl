#!/usr/bin/env python3
"""
Convert beyondmimic npz motion data to sonic pickle format

beyondmimic format (npz):
- joint_pos: (timesteps, 29)
- joint_vel: (timesteps, 29)
- body_pos_w: (timesteps, 35, 3)
- body_quat_w: (timesteps, 35, 4)
- body_lin_vel_w: (timesteps, 35, 3)
- body_ang_vel_w: (timesteps, 35, 3)

sonic format (joblib pickle):
- joint_pos: (timesteps, 29)
- joint_vel: (timesteps, 29)
- body_pos_w: (timesteps, 14, 3)
- body_quat_w: (timesteps, 14, 4)
- body_lin_vel_w: (timesteps, 14, 3)
- body_ang_vel_w: (timesteps, 14, 3)

This script maps the 35 SMPL body parts to 14 key body parts used by sonic.
"""

import argparse
import numpy as np
import joblib
import os

# Sonic 14 body parts mapping from SMPL 35 body parts
# Based on SMPL joint ordering:
# 0: pelvis, 1: left_hip, 2: right_hip, 3: spine1, 4: left_knee, 5: right_knee,
# 6: spine2, 7: left_ankle, 8: right_ankle, 9: spine3, 10: left_foot, 11: right_foot,
# 12: neck, 13: left_collar, 14: right_collar, 15: head,
# 16: left_shoulder, 17: right_shoulder, 18: left_elbow, 19: right_elbow,
# 20: left_wrist, 21: right_wrist, 22: left_hand, 23: right_hand
# ... plus additional SMPLX hand/finger joints (24-34)

# Map from beyondmimic 35 body parts to sonic 14 body parts
# This mapping is based on standard SMPL topology
SONIC_BODY_MAP = {
    0: 0,   # pelvis -> body_0 (pelvis)
    1: 1,   # left_hip -> body_1 (left_hip)
    2: 4,   # right_hip -> body_4 (right_hip)
    3: 3,   # spine1 -> body_3 (spine - approximated)
    4: 2,   # left_knee -> body_2 (left_knee)
    5: 5,   # right_knee -> body_5 (right_knee)
    6: 3,   # spine2 -> body_3 (spine - use spine)
    7: 6,   # left_ankle -> body_6 (left_ankle)
    8: 7,   # right_ankle -> body_7 (right_ankle)
    9: 3,   # spine3 -> body_3 (spine - use spine)
    10: 3,  # left_foot -> body_3 (approximated with spine for now)
    11: 3,  # right_foot -> body_3 (approximated with spine for now)
    12: 7,  # neck -> body_7 (head/neck)
    13: 7,  # left_collar -> body_7 (approximated with head)
    14: 7,  # right_collar -> body_7 (approximated with head)
    15: 7,  # head -> body_7 (head)
    16: 8,  # left_shoulder -> body_8 (left_shoulder)
    17: 11, # right_shoulder -> body_11 (right_shoulder)
    18: 9,  # left_elbow -> body_9 (left_elbow)
    19: 12, # right_elbow -> body_12 (right_elbow)
    20: 10, # left_wrist -> body_10 (left_wrist)
    21: 13, # right_wrist -> body_13 (right_wrist)
    22: 10, # left_hand -> body_10 (left_wrist)
    23: 13, # right_hand -> body_13 (right_wrist)
}

# Alternative mapping based on Isaac Lab G1 robot body parts
# G1 has 35 rigid bodies from URDF
# The first 14 are typically: pelvis, hip joints, knee joints, ankle joints, spine, head, shoulders, elbows, wrists
G1_BODY_MAP = {
    # These indices need to be verified based on the actual G1 URDF body ordering
    # This is a simplified mapping - you may need to adjust based on your URDF
    0: 0,   # pelvis
    1: 1,   # left_hip_yaw
    2: 4,   # right_hip_yaw
    3: 3,   # spine
    4: 2,   # left_knee
    5: 5,   # right_knee
    6: 7,   # left_ankle
    7: 6,   # right_ankle
    8: 7,   # neck/head (use right_ankle temporarily)
    9: 8,   # left_shoulder
    10: 11, # right_shoulder
    11: 9,  # left_elbow
    12: 12, # right_elbow
    13: 10, # left_wrist
    14: 13, # right_wrist
}

def map_body_parts(data_35, body_map, dtype='pos'):
    """Map 35 body parts to 14 body parts"""
    timesteps = data_35.shape[0]

    if dtype == 'pos':
        # data_35: (timesteps, 35, 3)
        result = np.zeros((timesteps, 14, 3), dtype=np.float32)
    elif dtype == 'quat':
        # data_35: (timesteps, 35, 4)
        result = np.zeros((timesteps, 14, 4), dtype=np.float32)
    elif dtype in ('lin_vel', 'ang_vel'):
        # data_35: (timesteps, 35, 3)
        result = np.zeros((timesteps, 14, 3), dtype=np.float32)

    for src_idx, dst_idx in body_map.items():
        if src_idx < data_35.shape[1]:
            result[:, dst_idx] = data_35[:, src_idx]

    return result


def convert_beyondmimic_to_sonic(input_npz, output_pkl, motion_name='motion'):
    """Convert beyondmimic npz to sonic pickle format"""

    print(f"Loading beyondmimic data from: {input_npz}")
    data = np.load(input_npz)

    print(f"  Keys: {list(data.keys())}")

    # Get dimensions
    timesteps = data['joint_pos'].shape[0]
    print(f"  Timesteps: {timesteps}")

    # Check body parts count
    body_parts = data['body_pos_w'].shape[1]
    print(f"  Body parts: {body_parts}")

    # Determine which mapping to use based on body parts count
    if body_parts == 35:
        body_map = SONIC_BODY_MAP
        print("  Using SMPL 35-to-14 mapping")
    else:
        print(f"  WARNING: Unexpected body parts count {body_parts}, using G1 mapping")
        body_map = G1_BODY_MAP

    # Create output dictionary
    motion_data = {}

    # Joint data (unchanged - 29 joints)
    motion_data['joint_pos'] = data['joint_pos'].astype(np.float32)
    motion_data['joint_vel'] = data['joint_vel'].astype(np.float32)

    # Map body positions (35 -> 14)
    motion_data['body_pos_w'] = map_body_parts(data['body_pos_w'], body_map, 'pos')
    motion_data['body_quat_w'] = map_body_parts(data['body_quat_w'], body_map, 'quat')
    motion_data['body_lin_vel_w'] = map_body_parts(data['body_lin_vel_w'], body_map, 'lin_vel')
    motion_data['body_ang_vel_w'] = map_body_parts(data['body_ang_vel_w'], body_map, 'ang_vel')

    # Add metadata
    motion_data['_body_indexes'] = {f'body_{i}': body_map.get(i, -1) for i in range(14)}
    motion_data['time_step_total'] = timesteps

    # Create output dictionary with motion name
    output_data = {motion_name: motion_data}

    # Save as joblib pickle
    print(f"Saving to: {output_pkl}")
    joblib.dump(output_data, output_pkl)
    print(f"  Successfully saved motion '{motion_name}' with {timesteps} timesteps")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert beyondmimic npz to sonic pickle format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input npz file (beyondmimic format)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output pkl file (sonic format)'
    )
    parser.add_argument(
        '--name', '-n',
        type=str,
        default='motion',
        help='Motion name in output (default: motion)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    convert_beyondmimic_to_sonic(args.input, args.output, args.name)

    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Run sonic converter: python gear_sonic_deploy/reference/convert_motions.py {args.output}")
    print(f"2. Or use the pkl directly in your sonic deployment")


if __name__ == "__main__":
    main()
