import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.transform import Rotation as R

# Function to parse the custom format data
def parse_data(file_path):
    timestamps = []
    positions = []
    orientations = []

    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Extract timestamp
                time_str = line.split('Time: ')[1].split(',')[0]
                timestamp = float(time_str)

                # Extract position (x, y, z) - we're interested in x, y
                position_str = line.split('Position: ')[1].split('),')[0].strip('()')
                position = [float(coord) for coord in position_str.split(', ')[:2]]

                # Extract orientation (quaternion)
                orientation_str = line.split('Orientation: ')[1].strip('()\n')
                quaternion = [float(q) for q in orientation_str.split(', ')]

                # Convert quaternion to yaw (theta)
                r = R.from_quat(quaternion)
                yaw = r.as_euler('zyx', degrees=False)[0]  # Extract yaw

                timestamps.append(timestamp)
                positions.append(position)
                orientations.append(yaw)
            except (IndexError, ValueError) as e:
                print(f"Skipping line due to parsing error: {line.strip()} - Error: {e}")
                continue

    return np.array(timestamps), np.array(positions), np.array(orientations)

# Parse the data
amcl_timestamps, amcl_positions, amcl_orientations = parse_data('/home/sashitsharma/Desktop/thesis_github/Thesis/Localization/datas/gmcl_pose_wo_odom_dynamic.txt')
gt_timestamps, gt_positions, gt_orientations = parse_data('/home/sashitsharma/Desktop/thesis_github/Thesis/Localization/datas/base_pose_ground_truth_gmcl_wo_odom_dynamic.txt')

# Synchronize data by matching timestamps exactly
def synchronize_data(gt_timestamps, gt_positions, gt_orientations, amcl_timestamps, amcl_positions, amcl_orientations):
    synchronized_gt_positions = []
    synchronized_gt_orientations = []
    synchronized_amcl_positions = []
    synchronized_amcl_orientations = []

    amcl_index = 0
    for i, gt_time in enumerate(gt_timestamps):
        while amcl_index < len(amcl_timestamps) and amcl_timestamps[amcl_index] < gt_time:
            amcl_index += 1
        if amcl_index < len(amcl_timestamps) and np.isclose(gt_time, amcl_timestamps[amcl_index]):
            synchronized_gt_positions.append(gt_positions[i])
            synchronized_gt_orientations.append(gt_orientations[i])
            synchronized_amcl_positions.append(amcl_positions[amcl_index])
            synchronized_amcl_orientations.append(amcl_orientations[amcl_index])

    return (np.array(synchronized_gt_positions), np.array(synchronized_gt_orientations),
            np.array(synchronized_amcl_positions), np.array(synchronized_amcl_orientations))

(synchronized_gt_positions, synchronized_gt_orientations,
 synchronized_amcl_positions, synchronized_amcl_orientations) = synchronize_data(
    gt_timestamps, gt_positions, gt_orientations,
    amcl_timestamps, amcl_positions, amcl_orientations
)

# Plot the trajectories
plt.figure(figsize=(10, 6))
plt.plot(synchronized_gt_positions[:, 0], synchronized_gt_positions[:, 1], label='Ground Truth Trajectory', color='blue')
plt.plot(synchronized_amcl_positions[:, 0], synchronized_amcl_positions[:, 1], label='GMCL Estimated Trajectory', color='red')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True)
plt.savefig('trajectory_comparison_gmcl_dynamic.png', dpi=300)  # Save the figure
plt.show()

# Calculate APE (Absolute Pose Error)
position_errors = np.sqrt((synchronized_amcl_positions[:, 0] - synchronized_gt_positions[:, 0])**2 + 
                          (synchronized_amcl_positions[:, 1] - synchronized_gt_positions[:, 1])**2)

orientation_errors = np.arctan2(np.sin(synchronized_amcl_orientations - synchronized_gt_orientations), 
                                np.cos(synchronized_amcl_orientations - synchronized_gt_orientations))

# Print APE values
#for i in range(len(position_errors)):
    #print(f"Time: {amcl_timestamps[i]:.2f}, Position Error: {position_errors[i]:.4f} meters, Orientation Error: {np.degrees(orientation_errors[i]):.4f} degrees")

# Calculate and print average APE values
mean_position_error = np.mean(position_errors)
mean_orientation_error = np.mean(np.degrees(orientation_errors))

print(f"Average Position Error (APE): {mean_position_error:.4f} meters")
print(f"Average Orientation Error (APE): {mean_orientation_error:.4f} degrees")

# Plot APE over time
plt.figure(figsize=(10, 6))
plt.plot(position_errors, label='Position Error (APE)', color='green')
plt.plot(np.degrees(orientation_errors), label='Orientation Error (degrees)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Absolute Pose Error (APE) Over Time')
plt.legend()
plt.grid(True)
plt.savefig('absolute_pose_error_gmcl_dynamic.png', dpi=300)  # Save the figure
plt.show()

# Calculate RMSE
rmse_position = np.sqrt(np.mean(position_errors**2))
rmse_orientation = np.sqrt(np.mean(orientation_errors**2))

# Print RMSE values
print(f'RMSE (Position): {rmse_position:.4f} meters')
print(f'RMSE (Orientation): {np.degrees(rmse_orientation):.4f} degrees')

# Calculate and Plot Running RMSE (Optional)
running_rmse_position = np.sqrt(np.cumsum(position_errors**2) / np.arange(1, len(position_errors) + 1))
running_rmse_orientation = np.sqrt(np.cumsum(orientation_errors**2) / np.arange(1, len(orientation_errors) + 1))

plt.figure(figsize=(10, 6))
plt.plot(running_rmse_position, label='Running RMSE (Position)', color='purple')
plt.plot(np.degrees(running_rmse_orientation), label='Running RMSE (Orientation)', color='brown')
plt.xlabel('Time Step')
plt.ylabel('RMSE')
plt.title('Running RMSE Over Time')
plt.legend()
plt.grid(True)
plt.savefig('running_rmse_gmcl_dynamic.png', dpi=300)  # Save the figure
plt.show()

# Calculate Hausdorff Distance
hausdorff_distance = max(directed_hausdorff(synchronized_gt_positions, synchronized_amcl_positions)[0], 
                         directed_hausdorff(synchronized_amcl_positions, synchronized_gt_positions)[0])

print(f'Hausdorff Distance: {hausdorff_distance:.4f} meters')
