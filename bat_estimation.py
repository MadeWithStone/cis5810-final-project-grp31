import numpy as np
import matplotlib.pyplot as plt
import trajectory

def vis_data_3d(kpts_3d, title=None):
    plt.figure(figsize=(4,4))
    ax = plt.axes(projection='3d')
    single_pose_index = [[5, 7, 9],
                                [6, 8, 10],
                                [5, 6, 12, 11, 5],
                                [11, 13, 15],
                                [12, 14, 16]]
    color_dict = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 3:'tab:red', 4:'tab:purple'}
    
    # Plot each fingers with same color
    for i, finger_index in enumerate(single_pose_index):
        curr_finger_kpts = np.array(kpts_3d[finger_index])
        ax.scatter(curr_finger_kpts[:,2], -curr_finger_kpts[:,0], curr_finger_kpts[:,1], color=color_dict[i])
        ax.plot3D(curr_finger_kpts[:,2], -curr_finger_kpts[:,0], curr_finger_kpts[:,1], color=color_dict[i])
    
    # Adjust 3D viewing angle as needed
    ax.view_init(elev=-90, azim=180, roll=0)
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

def convert_player_poses_to_meters(player_poses):
    shoulder1, shoulder2 = player_poses[0, 5], player_poses[0, 6]
    avg_width = 0.39 # m
    pixels = np.linalg.norm(shoulder1 - shoulder2)
    m_per_px = avg_width / pixels
    return player_poses * m_per_px


def get_bat_end_from_player_poses(player_poses):
    bat_length = 0.7 # m
    wrist1, wrist2 = player_poses[:, 9], player_poses[:, 10]
    wrists = (wrist1 + wrist2) / 2
    elbow1, elbow2 = player_poses[:, 7], player_poses[:, 8]
    forearm1 = wrists - elbow1
    forearm2 = wrists - elbow2
    
    bat = []
    for forearm1_pos, forearm2_pos, wrist_pos in zip(forearm1, forearm2, wrists):
        cross_prod = np.cross(forearm1_pos, forearm2_pos)
        bat.append(cross_prod / np.linalg.norm(cross_prod) * bat_length + wrist_pos)
    return np.array(bat)


def get_bat_force(bat_poses, fps=120,
                  m=0.9):
    ends_x = -bat_poses[:, 0]

    # find when bat is pointed most at camera (i.e. greatest x depth difference)
    num_poses = len(bat_poses)
    greatest_depth = 0.0
    j = 0
    for i in range(1, num_poses):
        x = ends_x[i]
        prev_x = ends_x[i - 1]
        depth = prev_x - x
        if depth > greatest_depth:
            greatest_depth = depth
            j = i

    # using greatest depth index, determine instantaneous acceleration
    v1 = (bat_poses[j] - bat_poses[j - 1]) / (1/fps) # m/s
    print("Bat Speed at Contact:\t", np.linalg.norm(v1), "m/s")
    a = v1 / (1/fps)
    F = m * a
    return F, j

def transform_bat_poses(bat_poses):
    T = np.array((
        (0, 0, -1),
        (-1, 0, 0),
        (0, -1, 0)
    ))
    for i in range(len(bat_poses)):
        bat_poses[i] = T @ bat_poses[i]
    return bat_poses
        
raw_player_poses = np.load("player_poses.npy") # wrists: 9 and 10, elbows: 7 and 8, shoulders: 5 and 6
player_poses = convert_player_poses_to_meters(raw_player_poses)
bat_poses = get_bat_end_from_player_poses(player_poses)
bat_poses = transform_bat_poses(bat_poses)
F, j = get_bat_force(bat_poses)
print("Force of Bat:\t\t", np.linalg.norm(F), "N")

trajectory.get_trajectory(F)
plt.show()
