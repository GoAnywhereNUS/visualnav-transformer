import os
import pickle
import shutil
import argparse
import numpy as np
from copy import deepcopy

MAX_SKIP_FRAMES = 3 # Taken from DECISION train script
LOGGING_FRAME_RATE = 15.0 # Frequency of logging in hz
CAMERA_FOLDER = "mid_color"

def main(args: argparse.Namespace):
    
    # Open the text file
    files = [f for f in os.listdir(args.input_dir) if "all" in f and f[-4:] == ".txt"]
    assert len(files) == 1
    
    with open(os.path.join(args.input_dir, files[0]), 'r') as f:
        lines = [line.split() for line in f]
        frame_ids = [int(line[0]) for line in lines[1:]]
        data = [line[2:] for line in lines[1:]] # Store: (linear_velocity, angular_velocity, intention)

    print(len(frame_ids), len(data))

    # List the trajectories
    split_trajectories = []
    traj = []
    prev_id = frame_ids[0] if len(frame_ids) > 0 else None
    last_frame_saved = -np.inf
    sample_time = 1.0 / args.sample_rate

    for frame_id, datum in zip(frame_ids, data):
        if frame_id - prev_id > MAX_SKIP_FRAMES:
            split_trajectories.append(traj)
            traj = []

        if float((frame_id - last_frame_saved)) / LOGGING_FRAME_RATE > sample_time:
            traj.append((frame_id, datum))
            prev_id = frame_id

        if len(split_trajectories) == 3:
            break

    print("Found", len(split_trajectories), "trajectories!")

    # Save each trajectory's data
    print("Writing data to file...")
    image_path = os.path.join(args.input_dir, CAMERA_FOLDER)
    for traj_i, traj in enumerate(split_trajectories):
        output_folder = os.path.join(args.output_dir, 't{0:05d}'.format(traj_i))
        os.mkdir(output_folder)
        traj_data = {
            "position": [], 
            "yaw": [], 
            "frame_ids": [], 
            "intentions": []
        }

        pos = np.zeros(3) # x, y, yaw (used for integrating only, discarded later)
        last_frame_id = None
        logging_frame_dt = 1.0 / LOGGING_FRAME_RATE
        for frame_i, (frame_id, datum) in enumerate(traj):
            shutil.copy(
                os.path.join(image_path, f'{frame_id}.jpg'),
                os.path.join(output_folder, f'{frame_i}.jpg')
            )

            v, w = float(datum[0]), float(datum[1])

            if last_frame_id is not None:
                pos += np.array([
                    v * np.cos(pos[2]) * logging_frame_dt,
                    v * np.sin(pos[2]) * logging_frame_dt,
                    w * logging_frame_dt
                ])

                # Angle bounds on yaw
                pos[2] = pos[2] % (2 * np.pi)

            traj_data["intentions"].append(datum[2])
            traj_data["frame_ids"].append(frame_id)
            traj_data["position"].append(deepcopy(pos[:2]))
            traj_data["yaw"].append(deepcopy(pos[2]))
            last_frame_id = frame_id

        with open(os.path.join(output_folder, 'traj_data.pkl'), 'wb') as f:
            pickle.dump(traj_data, f) 

        print("Wrote",  't{0:05d}'.format(traj_i), "of length:", len(traj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="name of the dataset (must be in process_config.yaml)",
        default="inet",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the datasets with rosbags",
        default="/data/home/joel/storage/inet_data",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="/data/home/joel/storage/inet_gnm_data",
        type=str,
        help="path for processed dataset (default: ../datasets/tartan_drive/)",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="sampling rate (default: 4.0 hz)",
    )

    args = parser.parse_args()
    main(args)