import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from scipy.interpolate import interp1d

def load_ground_truth_data(file_path):
    """
    Load the ground truth data from a text file.

    :param file_path: Path to the text file containing the ground truth data.
    :return: numpy array of ground truth points, shape (N, 2)
    """
    data = np.loadtxt(file_path, skiprows=1, usecols=(1, 2))
    return data

def load_slam_data(file_path):
    """
    Load the SLAM data from a text file.

    :param file_path: Path to the text file containing the SLAM data.
    :return: numpy array of SLAM points, shape (N, 2)
    """
    data = np.loadtxt(file_path, usecols=(0, 1))
    return data

def rotate_points(points, angle_degrees):
    """
    Rotate points by a given angle in degrees.

    :param points: numpy array of points, shape (N, 2)
    :param angle_degrees: The angle by which to rotate the points, in degrees.
    :return: Rotated points, shape (N, 2)
    """
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians),  np.cos(angle_radians)]])
    rotated_points = np.dot(points, rotation_matrix)
    return rotated_points

def translate_slam_data(slam_points, offset):
    """
    Translate the SLAM data by the given offset.

    :param slam_points: numpy array of SLAM points, shape (N, 2)
    :param offset: numpy array representing the translation vector, shape (2,)
    :return: Translated SLAM points, shape (N, 2)
    """
    return slam_points + offset

def synchronize_data(gt_points, slam_points):
    """
    Synchronize ground truth and SLAM data to have the same number of points
    by interpolating the ground truth data to match the SLAM data length.
    """
    n_slam = len(slam_points)
    gt_times = np.linspace(0, 1, len(gt_points))
    slam_times = np.linspace(0, 1, n_slam)
    
    interp_gt_x = interp1d(gt_times, gt_points[:, 0], kind='linear')
    interp_gt_y = interp1d(gt_times, gt_points[:, 1], kind='linear')
    
    synced_gt_points = np.vstack((interp_gt_x(slam_times), interp_gt_y(slam_times))).T
    return synced_gt_points, slam_points

def calculate_ate(gt_points, slam_points):
    """
    Calculate the Absolute Trajectory Error (ATE) between ground truth and SLAM points.
    """
    ate = np.sqrt(np.mean(np.sum((gt_points - slam_points) ** 2, axis=1)))
    return ate

def calculate_euclidean_distance(points):
    """
    Calculate the Euclidean distance between the first and last points in the trajectory.
    """
    start_point = points[0]
    end_point = points[-1]
    distance = np.sqrt(np.sum((end_point - start_point) ** 2))
    return distance

def calculate_rmse_position(gt_points, slam_points):
    """
    Calculate the RMSE of the position error between ground truth and SLAM points.
    """
    position_errors = np.sqrt(np.sum((gt_points - slam_points) ** 2, axis=1))
    rmse = np.sqrt(np.mean(position_errors ** 2))
    return rmse

def calculate_average_position_error(gt_points, slam_points):
    """
    Calculate the average position error between ground truth and SLAM points.
    """
    position_errors = np.sqrt(np.sum((gt_points - slam_points) ** 2, axis=1))
    average_position_error = np.mean(position_errors)
    return average_position_error

def calculate_hausdorff_distance(gt_points, slam_points):
    """
    Calculate the Hausdorff distance between ground truth and SLAM points.
    """
    hausdorff_distance = max(directed_hausdorff(gt_points, slam_points)[0], 
                             directed_hausdorff(slam_points, gt_points)[0])
    return hausdorff_distance

def plot_trajectories(gt_points, aligned_slam_points):
    """
    Plot the ground truth and aligned SLAM trajectories.

    :param gt_points: numpy array of ground truth points, shape (N, 2)
    :param aligned_slam_points: numpy array of aligned SLAM points, shape (N, 2)
    """
    plt.figure(figsize=(8, 6))  # Adjust the figure size if necessary
    plt.plot(gt_points[:, 0], gt_points[:, 1], 'b-', label='Ground Truth')
    plt.plot(aligned_slam_points[:, 0], aligned_slam_points[:, 1], 'g-', label='GMapping SLAM')
    plt.legend()
    plt.title('Ground Truth vs GMapping SLAM Trajectories')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')  # Ensure the same scale for x and y
    plt.xlim([-1.5, 1.5])  # Adjust x-axis limits to zoom in
    plt.ylim([-2.5, 2.5])  # Adjust y-axis limits to zoom in
    plt.grid(True)
    plt.savefig('trajectory_turtlebot_gmapping_slam.png', dpi=300)
    plt.show()

def main():
    # Load ground truth and SLAM data from text files
    gt_file_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Turtlebot3/ground_truth_pose.txt'  # Replace with your actual ground truth file path
    slam_file_path = '/home/sashitsharma/Desktop/thesis_github/Thesis/SLAM/Turtlebot3/gmapping/gmapping_pose.txt'        # Replace with your actual SLAM file path

    gt_points = load_ground_truth_data(gt_file_path)
    slam_points = load_slam_data(slam_file_path)

    # Rotate the ground truth points by 90 degrees to the left
    rotated_gt_points = rotate_points(gt_points, 270)

    # Known offset between SLAM start and ground truth start
    offset = np.array([-0.0321, -1.9389])
    
    # Apply the translation to the SLAM points
    aligned_slam_points = translate_slam_data(slam_points, offset)

    # Synchronize the data points
    synced_gt_points, synced_slam_points = synchronize_data(rotated_gt_points, aligned_slam_points)

    # Plot the trajectories with rotated ground truth points
    plot_trajectories(synced_gt_points, synced_slam_points)

    # Calculate metrics
    ate = calculate_ate(synced_gt_points, synced_slam_points)
    print(f"Absolute Trajectory Error (ATE): {ate:.4f} meters")

    gt_distance = calculate_euclidean_distance(synced_gt_points)
    slam_distance = calculate_euclidean_distance(synced_slam_points)
    print(f"Ground Truth Distance (start to end): {gt_distance:.4f} meters")
    print(f"SLAM Distance (start to end): {slam_distance:.4f} meters")

    rmse_position = calculate_rmse_position(synced_gt_points, synced_slam_points)
    print(f"RMSE (Position Error): {rmse_position:.4f} meters")

    average_position_error = calculate_average_position_error(synced_gt_points, synced_slam_points)
    print(f"Average Position Error: {average_position_error:.4f} meters")

    hausdorff_distance = calculate_hausdorff_distance(synced_gt_points, synced_slam_points)
    print(f"Hausdorff Distance: {hausdorff_distance:.4f} meters")

if __name__ == "__main__":
    main()
