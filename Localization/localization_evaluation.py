import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial.transform import Rotation as R

# Load data
# Assuming the data is stored in CSV files with columns: timestamp, x, y, theta
amcl_data = np.loadtxt('amcl_pose.csv', delimiter=',')
ground_truth_data = np.loadtxt('ground_truth.csv', delimiter=',')

# Synchronize data by timestamps
# Assuming that both datasets have the same or close timestamps, you can do:
def synchronize_data(gt_data, amcl_data):
    synchronized_amcl = []
    for gt in gt_data:
        closest_amcl = min(amcl_data, key=lambda x: abs(x[0] - gt[0]))
        synchronized_amcl.append(closest_amcl)
    return np.array(synchronized_amcl)

synchronized_amcl_data = synchronize_data(ground_truth_data, amcl_data)

# Extract positions and orientations
gt_positions = ground_truth_data[:, 1:3]
amcl_positions = synchronized_amcl_data[:, 1:3]

gt_orientations = ground_truth_data[:, 3]
amcl_orientations = synchronized_amcl_data[:, 3]

# Plot the trajectories
plt.figure(figsize=(10, 6))
plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth Trajectory', color='blue')
plt.plot(amcl_positions[:, 0], amcl_positions[:, 1], label='AMCL Estimated Trajectory', color='red')
plt.xlabel('X Position (meters)')
plt.ylabel('Y Position (meters)')
plt.title('Trajectory Comparison')
plt.legend()
plt.grid(True)
plt.show()

# Calculate APE (Absolute Pose Error)
position_errors = np.sqrt((amcl_positions[:, 0] - gt_positions[:, 0])**2 + 
                          (amcl_positions[:, 1] - gt_positions[:, 1])**2)

orientation_errors = np.arctan2(np.sin(amcl_orientations - gt_orientations), 
                                np.cos(amcl_orientations - gt_orientations))

# Plot APE over time
plt.figure(figsize=(10, 6))
plt.plot(position_errors, label='Position Error (APE)', color='green')
plt.plot(np.degrees(orientation_errors), label='Orientation Error (degrees)', color='orange')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Absolute Pose Error (APE) Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
rmse_position = np.sqrt(np.mean(position_errors**2))
rmse_orientation = np.sqrt(np.mean(orientation_errors**2))

print(f'RMSE (Position): {rmse_position:.4f} meters')
print(f'RMSE (Orientation): {np.degrees(rmse_orientation):.4f} degrees')

# Calculate Hausdorff Distance
hausdorff_distance = max(directed_hausdorff(gt_positions, amcl_positions)[0], 
                         directed_hausdorff(amcl_positions, gt_positions)[0])

print(f'Hausdorff Distance: {hausdorff_distance:.4f} meters')

