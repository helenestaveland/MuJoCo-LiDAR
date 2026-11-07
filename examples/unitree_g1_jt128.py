"""
Unitree G1 simulation with Hesai JT128 LiDAR and ROS2 integration.

Built using the MuJoCo-LiDAR framework and its existing
Unitree G1 / ROS2 examples:
https://github.com/discoverse-dev/MuJoCo-LiDAR
Reference: examples/unitree_g1_ros2.py

JT128-specific adaptations include the beam-angle scanning pattern,
Hesai-compatible ROS2 PointCloud2 fields, IMU publishing, and
ground-truth odometry publishing.

LiDAR raycasting is provided by the original MuJoCo-LiDAR library.
JT128 beam-angle calibration data originates from Hesai.
"""
import argparse
import array
import time
import sys

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as rt
from etils import epath

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from nav_msgs.msg import Odometry

from mujoco_lidar import MjLidarWrapper
from jt128_scan_test import generate_jt128

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"
_MJCF_PATH = _HERE.parent / "models" / "scene_g1.xml"

_JOINT_NUM = 29


class JT128Publisher(Node):
    def __init__(self):
        super().__init__("jt128_mujoco_publisher")
        self.publisher = self.create_publisher(
            PointCloud2,
            "/rslidar_points",
            10,
        )
        self.imu_publisher = self.create_publisher(
            Imu,
            "/vectornav_driver_node/imu/data",
            50,
        )
        self.gt_publisher = self.create_publisher(
            Odometry,
            "/mujoco/ground_truth",
            10,
        )

        # Map MuJoCo simulation time onto the current ROS clock.
        # Both LiDAR and IMU then use the same clock domain.
        self._ros_time_origin_ns = self.get_clock().now().nanoseconds
        self._last_publish_wall = None

    def publish_points(self, points, sim_time):
        now = time.perf_counter()
        if self._last_publish_wall is not None:
            dt = now - self._last_publish_wall
            print(f"  SOURCE interval: {dt*1000:.1f} ms ({1.0/dt:.2f} Hz)")
        self._last_publish_wall = now

        points = np.asarray(points, dtype=np.float32)

        # MuJoCo-LiDAR represents rays with no hit as (0, 0, 0).
        # Do not publish those as real LiDAR returns.
        ranges_sq = np.einsum("ij,ij->i", points, points)
        valid = ranges_sq > 1e-12

        original_indices = np.nonzero(valid)[0]
        points = points[valid]

        header = self._header(sim_time, "mimosa_lidar")
        header_ts = (
            float(header.stamp.sec)
            + float(header.stamp.nanosec) * 1e-9
        )

        # Hesai-compatible point layout expected by MIMOSA.
        # 32-byte stride leaves padding after ring.
        point_dtype = np.dtype({
            "names": [
                "x",
                "y",
                "z",
                "intensity",
                "timestamp",
                "ring",
            ],
            "formats": [
                "<f4",
                "<f4",
                "<f4",
                "<f4",
                "<f8",
                "<u2",
            ],
            "offsets": [
                0,
                4,
                8,
                12,
                16,
                24,
            ],
            "itemsize": 32,
        })

        cloud = np.empty(points.shape[0], dtype=point_dtype)

        cloud["x"] = points[:, 0]
        cloud["y"] = points[:, 1]
        cloud["z"] = points[:, 2]

        # MuJoCo currently has no reflectivity model.
        cloud["intensity"] = 1.0

        # Current simulation casts the entire scan from one pose,
        # so all points have the same acquisition time.
        cloud["timestamp"] = header_ts

        # Flattening order is:
        # horizontal sample 0: rings 0..127,
        # horizontal sample 1: rings 0..127, etc.
        cloud["ring"] = (original_indices % 128).astype(np.uint16)

        from sensor_msgs.msg import PointField

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = cloud.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = 32
        msg.row_step = msg.point_step * msg.width

        msg.fields = [
            PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="timestamp", offset=16, datatype=PointField.FLOAT64, count=1),
            PointField(name="ring",      offset=24, datatype=PointField.UINT16,  count=1),
        ]

        # array.array('B') uses the fast ROS 2 Humble setter path.
        msg.data = array.array("B", cloud.tobytes())

        self.publisher.publish(msg)
    def _stamp_from_sim_time(self, sim_time):
        from builtin_interfaces.msg import Time

        total_ns = self._ros_time_origin_ns + int(round(float(sim_time) * 1e9))

        stamp = Time()
        stamp.sec = total_ns // 1_000_000_000
        stamp.nanosec = total_ns % 1_000_000_000
        return stamp

    def _header(self, sim_time, frame_id):
        from std_msgs.msg import Header

        header = Header()
        header.stamp = self._stamp_from_sim_time(sim_time)
        header.frame_id = frame_id
        return header

    def publish_imu(self, model, data):
        msg = Imu()
        msg.header = self._header(data.time, "mimosa_imu")

        # IMU sensors from the Unitree MuJoCo G1 model.
        gyro = data.sensor("imu_gyro").data
        accel = data.sensor("imu_acc").data

        msg.angular_velocity.x = float(gyro[0])
        msg.angular_velocity.y = float(gyro[1])
        msg.angular_velocity.z = float(gyro[2])

        msg.linear_acceleration.x = float(accel[0])
        msg.linear_acceleration.y = float(accel[1])
        msg.linear_acceleration.z = float(accel[2])

        # MIMOSA does not use sensor_msgs/Imu.orientation.
        # -1 indicates that orientation is not provided.
        msg.orientation_covariance[0] = -1.0

        self.imu_publisher.publish(msg)

    def publish_ground_truth(self, model, data):
        msg = Odometry()
        msg.header = self._header(data.time, "mujoco_world")
        msg.child_frame_id = "mujoco_torso_imu"

        site_id = model.site("imu").id
        pos = data.site_xpos[site_id]

        # MuJoCo framequat sensor order is [w, x, y, z].
        quat = data.sensor("imu_quat").data

        linvel = data.sensor("frame_vel").data
        angvel = data.sensor("imu_gyro").data

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        msg.pose.pose.orientation.w = float(quat[0])
        msg.pose.pose.orientation.x = float(quat[1])
        msg.pose.pose.orientation.y = float(quat[2])
        msg.pose.pose.orientation.z = float(quat[3])

        msg.twist.twist.linear.x = float(linvel[0])
        msg.twist.twist.linear.y = float(linvel[1])
        msg.twist.twist.linear.z = float(linvel[2])

        msg.twist.twist.angular.x = float(angvel[0])
        msg.twist.twist.angular.y = float(angvel[1])
        msg.twist.twist.angular.z = float(angvel[2])

        self.gt_publisher.publish(msg)


class OnnxController:
    """ONNX controller for the G-1 robot."""

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        policy_path: str,
        default_angles: np.ndarray,
        ctrl_dt: float,
        n_substeps: int,
        action_scale: float = 0.5,
        lidar_type: str = "mid360",
        backend: str = "jax",
    ):

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self._phase = np.array([0.0, np.pi])
        self._gait_freq = 1.5
        self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])

        # JT128 LiDAR
        self.dynamic_lidar = False
        self.rays_theta, self.rays_phi = generate_jt128()

        self.rays_theta = np.ascontiguousarray(self.rays_theta).astype(np.float32)
        self.rays_phi = np.ascontiguousarray(self.rays_phi).astype(np.float32)

        # LiDAR should ray-cast against the G1 collision geometry,
        # not the detailed visual meshes.
        geomgroup = np.zeros((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[1] = 1
        self.lidar = MjLidarWrapper(
            mj_model,
            site_name="lidar",
            backend=backend,
            args={"bodyexclude": mj_model.body("torso_link").id, "geomgroup": geomgroup},
        )

    def get_obs(self, model, data) -> np.ndarray:
        linvel = data.sensor("local_linvel_pelvis").data
        gyro = data.sensor("gyro_pelvis").data
        imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7 : 7 + _JOINT_NUM] - self._default_angles
        joint_velocities = data.qvel[6 : 6 + _JOINT_NUM]
        phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
        command = np.zeros(3, dtype=np.float32)
        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity,
                command,
                joint_angles,
                joint_velocities,
                self._last_action,
                phase,
            ]
        )
        return obs.astype(np.float32)

    def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(model, data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            self._last_action = onnx_pred.copy()
            data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
            if data.time >= 15.0:
                phase_tp1 = self._phase + self._phase_dt
                self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuJoCo LiDAR可视化与Unitree G1 ROS2集成")
    parser.add_argument(
        "--backend",
        type=str,
        default="jax",
        help="LiDAR后端 (cpu, taichi, jax, warp)",
        choices=["cpu", "taichi", "jax", "warp"],
    )
    args = parser.parse_args()

    rclpy.init()
    ros_node = JT128Publisher()

    mj_model = mujoco.MjModel.from_xml_path(_MJCF_PATH.as_posix())
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

    ctrl_dt = 0.02
    lidar_dt = 1.0 / 10.0
    mj_model.opt.timestep = 0.004

    policy = OnnxController(
        mj_model,
        policy_path=(_ONNX_DIR / "g1_policy.onnx").as_posix(),
        default_angles=np.array(mj_model.keyframe("home").qpos[7 : 7 + _JOINT_NUM]),
        ctrl_dt=ctrl_dt,
        n_substeps=int(round(ctrl_dt / mj_model.opt.timestep)),
        action_scale=0.5,
        backend=args.backend,
    )

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # The LiDAR still uses all JT128 rays, but only display as many
        # points as the MuJoCo viewer can hold.
        # Display only a subset of the 115,200 rays.
        # The LiDAR itself still traces all 115,200 rays.
        display_count = 0
        display_indices = np.linspace(
            0, policy.rays_theta.shape[0] - 1, display_count, dtype=int
        )

        viewer.user_scn.ngeom = display_count
        for i in range(viewer.user_scn.ngeom):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.01, 0, 0],
                pos=[0, 0, 0],
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 0.8]),
            )
        print("Starting simulation...")
        print("Number of rays:", policy.rays_theta.shape[0])

        # 创建颜色映射
        cmap = plt.get_cmap("hsv")  # 或使用 'jet', 'viridis', 'plasma' 等

        _last_time = 1e6
        n_substeps = int(round(lidar_dt / mj_model.opt.timestep))

        step_acc = 0.0
        control_acc = 0.0
        sync_acc = 0.0
        sleep_acc = 0.0

        while viewer.is_running():
            if mj_data.time < _last_time:
                _counter = 0
                _start_time = time.time()
            _last_time = mj_data.time

            t_part = time.perf_counter()
            mujoco.mj_step(mj_model, mj_data)
            step_acc += time.perf_counter() - t_part

            # Publish the torso IMU and matching MuJoCo ground truth.
            ros_node.publish_imu(mj_model, mj_data)
            ros_node.publish_ground_truth(mj_model, mj_data)

            t_part = time.perf_counter()
            policy.get_control(mj_model, mj_data)
            control_acc += time.perf_counter() - t_part

            _counter += 1
            if _counter % n_substeps == 0:
                if policy.dynamic_lidar:
                    policy.rays_theta, policy.rays_phi = policy.livox_generator.sample_ray_angles()
                # Time the JT128 ray casting.
                t_lidar = time.perf_counter()
                policy.lidar.trace_rays(mj_data, policy.rays_theta, policy.rays_phi)
                points = policy.lidar.get_hit_points()

                ranges = np.linalg.norm(points, axis=1)
                valid = ranges > 1e-6
                if np.any(valid):
                    print(
                        f"  VALID RETURNS: {valid.sum()}/{len(points)} | "
                        f"range {ranges[valid].min():.2f}–{ranges[valid].max():.2f} m"
                    )
                else:
                    print("  VALID RETURNS: 0")

                world_points = (
                    points @ policy.lidar.sensor_rotation.T + policy.lidar.sensor_position
                )
                lidar_time = time.perf_counter() - t_lidar

                # Time PointCloud2 construction + publish separately from spin_once.
                t_publish = time.perf_counter()
                ros_node.publish_points(points, mj_data.time)
                publish_time = time.perf_counter() - t_publish

                t_spin = time.perf_counter()
                rclpy.spin_once(ros_node, timeout_sec=0.0)
                spin_time = time.perf_counter() - t_spin

                print(
                    f"LiDAR: {lidar_time*1000:.1f} ms | "
                    f"PointCloud2 + publish: {publish_time*1000:.1f} ms | "
                    f"spin_once: {spin_time*1000:.1f} ms"
                )

                # 根据高度设置颜色
                z_values = world_points[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                if z_max > z_min:
                    # 归一化高度值到 [0, 1]
                    z_norm = (z_values - z_min) / (z_max - z_min)
                else:
                    z_norm = np.zeros_like(z_values)

                # 使用 matplotlib 颜色映射
                colors = cmap(z_norm)  # 返回 RGBA 值，shape: (N, 4)

                display_points = world_points[display_indices]
                display_colors = colors[display_indices]

                for i in range(viewer.user_scn.ngeom):
                    viewer.user_scn.geoms[i].pos[:] = display_points[i]
                    viewer.user_scn.geoms[i].rgba[:] = display_colors[i]

            if _counter % 5 == 0:
                t_part = time.perf_counter()
                viewer.sync()
                sync_acc += time.perf_counter() - t_part

            run_time = time.time() - _start_time
            if run_time < mj_data.time:
                t_part = time.perf_counter()
                time.sleep(mj_data.time - run_time)
                sleep_acc += time.perf_counter() - t_part

            if _counter % n_substeps == 0:
                print(
                    f"  25-step totals: mj_step={step_acc*1000:.1f} ms | "
                    f"control={control_acc*1000:.1f} ms | "
                    f"viewer.sync={sync_acc*1000:.1f} ms | "
                    f"sleep={sleep_acc*1000:.1f} ms"
                )
                step_acc = 0.0
                control_acc = 0.0
                sync_acc = 0.0
                sleep_acc = 0.0

    ros_node.destroy_node()
    rclpy.shutdown()
