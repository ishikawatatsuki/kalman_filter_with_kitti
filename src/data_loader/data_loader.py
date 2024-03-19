import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pykitti
from ahrs.filters import AngularRate

from utils import normalize_angles, lla_to_enu, get_rigid_transformation

sequence_data_map = {
    '0033': {
        'kitti_date': '2011_09_30',
        'calib_velo_to_cam': '2011_09_30/calib_velo_to_cam.txt',
        'calib_imu_to_velo': '2011_09_30/calib_imu_to_velo.txt',
        'vo_path': 'trajectory_estimated_09.npy',
        'gt_path': 'trajectory_gt_09.npy',
        'imu_path': 'imu_09.npy'
    }
}

class DataLoader:

    N = None
    ts = None
    dataset = None
    T_from_imu_to_cam = None
    T_from_cam_to_imu = None

    GPS_measurement_noise_std = 1.0
    VO_noise_std = 1.0
    IMU_acc_noise_std = 0.02
    IMU_angular_velocity_noise_std = 0.01 # standard deviation of yaw rate in rad/s
    velocity_noise_std = 0.3
    
    GPS_measurements_in_meter = []  # [longitude(deg), latitude(deg), altitude(meter)] x N from GPS
    VO_measurements = [] # [longitude(deg), latitude(deg)] x N from Visual Odometry
    IMU_outputs = [] # [acc_x, acc_y, acc_z, ang_vel_x, ang_vel_y, ang_vel_z] x N from IMU
    INS_angles = [] # [roll(rad), pitch(rad), yaw(rad)] x N
    INS_velocities = [] # [forward velocity, leftward velocity, upward velocity] x N from INS

    IMU_quaterion = None

    GPS_mesurement_in_meter_with_noise = None
    VO_measurements_with_noise = None
    IMU_acc_with_noise = None
    IMU_angular_velocity_with_noise = None
    INS_velocities_with_noise = None

    def __init__(self, sequence_nr='0033', kitti_root_dir='../data', vo_root_dir='../vo_estimates'):
        
        self.config = sequence_data_map[sequence_nr]
        self.config['calib_velo_to_cam'] = kitti_root_dir + '/' + self.config['calib_velo_to_cam']
        self.config['calib_imu_to_velo'] = kitti_root_dir + '/' + self.config['calib_imu_to_velo']
        self.config['vo_path'] = vo_root_dir + '/' + self.config['vo_path']
        self.config['gt_path'] = vo_root_dir + '/' + self.config['gt_path']
        self.config['imu_path'] = vo_root_dir + '/' + self.config['imu_path']
        
        self.dataset = pykitti.raw(kitti_root_dir, self.config['kitti_date'], sequence_nr)
        
        print("Loading calibration files.")
        T_velo_ref0 = get_rigid_transformation(self.config['calib_velo_to_cam'])
        T_imu_velo = get_rigid_transformation(self.config['calib_imu_to_velo'])
        self.T_from_imu_to_cam = T_imu_velo @ T_velo_ref0
        self.T_from_cam_to_imu = np.linalg.inv(self.T_from_imu_to_cam)
        
        self.load_data()
        self.report()
        self.show_trajectory()
        self.add_noise()

        
    def load_data(self):
        imus = []
        ins_w = []
        ins_v = []
        vo = np.load(self.config['vo_path'])
        gt = np.load(self.config['gt_path'])
        self.N = gt.shape[1]
        for index, oxts_data in enumerate(self.dataset.oxts):
            if index < self.N:
                packet = oxts_data.packet
                # GPS_measurements.append([
                #     packet.lon,
                #     packet.lat,
                #     packet.alt
                # ])
                imus.append([
                    packet.ax,
                    packet.ay,
                    packet.az,
                    packet.wx,
                    packet.wy,
                    packet.wz
                ])
                ins_w.append([
                    packet.roll,
                    packet.pitch,
                    packet.yaw
                ])
                ins_v.append([
                    packet.vf,
                    packet.vl,
                    packet.vu
                ])
            
        self.GPS_measurements_in_meter = self.transform_gps_data(np.array(gt).T)
        self.VO_measurements = self.transform_vo_data(np.array(vo[:self.N]))
        self.IMU_outputs = np.array(imus)
        self.INS_angles = np.array(ins_w)
        self.INS_velocities = np.array(ins_v)
        
        timestamps = np.array(self.dataset.timestamps[:self.N])
        elapsed = np.array(timestamps) - timestamps[0]
        self.ts = [t.total_seconds() for t in elapsed] # dt
        
    def transform_gps_data(self, gps_data):
        print("Transform GPS data into imu coordinate.")
        GPS_measurements_in_meter = []
        for gt_est in gps_data:
            lla_values = np.array([gt_est[0], gt_est[1], gt_est[2], 1])
            transformed_lla_values = self.T_from_cam_to_imu @ lla_values
            GPS_measurements_in_meter.append([transformed_lla_values[0], 
                                     transformed_lla_values[1], 
                                     transformed_lla_values[2]])
        
        return np.array(GPS_measurements_in_meter)

    def transform_vo_data(self, vo_data):
        print("Transform VO data into imu coordinate.")
        vo_array = []
        for vo_est in vo_data:
            VO = np.array([vo_est[0], vo_est[1], vo_est[2], 1])
            transformed = self.T_from_cam_to_imu @ VO
            vo_array.append([transformed[0], transformed[1], transformed[2]])
        
        return np.array(vo_array)

    def report(self):
        print(f"Data size: {self.N}")
        print("Shape:")
        print(f'GPS: {self.GPS_measurements_in_meter.shape}')
        print(f'VO: {self.VO_measurements.shape}')
        print(f'IMU: {self.IMU_outputs.shape}')
        print(f'INS angle: {self.INS_angles.shape}')
        print(f'INS velocity: {self.INS_velocities.shape}')

    def show_trajectory(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        xs, ys, _ = self.GPS_measurements_in_meter.T
        ax.plot(xs, ys, label='ground-truth trajectory (GPS)')
        xs, ys, _ = self.VO_measurements.T
        ax.plot(xs, ys, label='Visual odometry')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()

    def add_noise(self, std=1.0):
        print("Add noise to GPS data")
        
        _gps_noise = np.random.normal(0.0, self.GPS_measurement_noise_std, (self.N, 2))  # gen gaussian noise
        self.GPS_mesurement_in_meter_with_noise = self.GPS_measurements_in_meter.copy()
        self.GPS_mesurement_in_meter_with_noise[:, :2] += _gps_noise  # add the noise to ground-truth x and y positions

        print("Adding noise to VO data")
        _vo_noise = np.random.normal(0.0, self.VO_noise_std, (self.N, 2))  # gen gaussian noise
        self.VO_measurements_with_noise = self.VO_measurements.copy()
        self.VO_measurements_with_noise[:, :2] += _vo_noise  # add the noise to ground-truth x and y positions
        print("Adding noise to IMU sensor data")
        print("Adding noise to linear acceleration")
        IMU_acc_noise = np.random.normal(0.0, self.IMU_acc_noise_std,(self.N, 3))  # gaussian noise
        self.IMU_acc_with_noise = self.IMU_outputs[:, :3].copy()
        self.IMU_acc_with_noise += IMU_acc_noise
        
        
        print("Adding noise to angular velocity")
        IMU_angular_velocity_noise = np.random.normal(0.0, self.IMU_angular_velocity_noise_std, (self.N,3))  # gen gaussian noise
        self.IMU_angular_velocity_with_noise = self.IMU_outputs[:, 3:].copy()
        self.IMU_angular_velocity_with_noise += IMU_angular_velocity_noise  # add the noise to angular velocity as measurement noise

        angular_rate = AngularRate(gyr=self.IMU_angular_velocity_with_noise)
        self.IMU_quaterion = angular_rate.Q

        print("Adding noise to INS sensor data")
        print("Adding noise to linear velocity data")
        velocity_noise = np.random.normal(0.0, self.velocity_noise_std, (self.N, 3))
        self.INS_velocities_with_noise = self.INS_velocities.copy()
        self.INS_velocities_with_noise += velocity_noise

    def show_vo_with_noise(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        xs, ys, _ = self.VO_measurements.T
        ax.plot(xs, ys, lw=2, label='VO trajectory')
        
        xs, ys, _ = self.VO_measurements_with_noise.T
        ax.plot(xs, ys, lw=0, marker='.', markersize=5, alpha=0.4, label='noised VO trajectory')
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid()

    def show_linear_acceleration_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        acc_y_labels = ['acceleration along x[m/s^2]', 'acceleration along y[m/s^2]', 'acceleration along z[m/s^2]']
        
        for idx in range(1, 4):  
            i = idx - 1
            ax[i].plot(self.ts, self.IMU_outputs[:, idx-1:idx], lw=1, label='ground-truth')
            ax[i].plot(self.ts, self.IMU_acc_with_noise[:, idx-1:idx], lw=0, marker='.', alpha=0.4, label='observed')
            ax[i].set_xlabel('time elapsed [sec]')
            ax[i].set_ylabel(acc_y_labels[i])
            ax[i].legend()
        fig.tight_layout()

    def show_angular_velocity_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        angualr_vel_y_labels = ['angualr velocity about x[rad/s]', 'angualr velocity about y[rad/s]', 'angualr velocity about z[rad/s]']
        
        for idx in range(3):  
            i = idx + 4
            ax[idx].plot(self.ts, self.IMU_outputs[:, i-1:i], lw=1, label='ground-truth')
            ax[idx].plot(self.ts, self.IMU_angular_velocity_with_noise[:, idx:idx+1], lw=0, marker='.', alpha=0.4, label='observed')
            ax[idx].set_xlabel('time elapsed [sec]')
            ax[idx].set_ylabel(angualr_vel_y_labels[idx])
            ax[idx].legend()
        fig.tight_layout()
    
    def show_linear_velocity_with_noise(self):
        fig, ax = plt.subplots(3, 1, figsize=(6, 9))
        linear_velocity_y_labels = ['linear velocity along x[m/s]', 'linear velocity along y[m/s]', 'alinear velocity along z[m/s]']
        
        for idx in range(1, 4):  
            i = idx - 1
            ax[i].plot(self.ts, self.INS_velocities[:, idx-1:idx], lw=1, label='ground-truth')
            ax[i].plot(self.ts, self.INS_velocities_with_noise[:, idx-1:idx], lw=0, marker='.', alpha=0.4, label='observed')
            ax[i].set_xlabel('time elapsed [sec]')
            ax[i].set_ylabel(linear_velocity_y_labels[i])
            ax[i].legend()
        fig.tight_layout()

if __name__ == "__main__":
    # os.chdir("../src")
    # from utils import normalize_angles, lla_to_enu, get_rigid_transformation
    
    data = DataLoader(kitti_root_dir="../../data", vo_root_dir="../../vo_estimates")
    













