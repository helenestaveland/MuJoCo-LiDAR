"""
Hesai JT128 scan-pattern adaptation for MuJoCo-LiDAR.

Based on the scan-generation approach provided by the
MuJoCo-LiDAR project:
https://github.com/discoverse-dev/MuJoCo-LiDAR

Original project: MuJoCo-LiDAR (v0.3.5)
Reference: src/mujoco_lidar/scan_gen.py

This adaptation uses Hesai JT128 per-channel elevation and
azimuth calibration angles to generate ray directions in
the format expected by MuJoCo-LiDAR.

Calibration data: Hesai JT128_default_angle.csv, downloaded
from Hesai's official website.
"""
import csv
import numpy as np


def generate_jt128(
    csv_path="JT128_default_angle.csv",
    horizontal_samples=900,
):
    elevations = []
    azimuth_offsets = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            elevations.append(float(row["Elevation"]))
            azimuth_offsets.append(float(row["Azimuth"]))

    elevations = np.deg2rad(np.asarray(elevations, dtype=np.float32))
    azimuth_offsets = np.deg2rad(np.asarray(azimuth_offsets, dtype=np.float32))

    horizontal = np.deg2rad(
        np.linspace(
            0.0,
            360.0,
            horizontal_samples,
            endpoint=False,
            dtype=np.float32,
        )
    )

    # Hesai JT128:
    #   0 deg azimuth = +Y
    #   positive azimuth rotates clockwise viewed from above
    #
    # MuJoCo-LiDAR:
    #   theta = 0 deg = +X
    #   positive theta -> +Y
    #
    # Therefore:
    #   theta_mujoco = 90 deg - theta_hesai

    theta = (
        np.pi / 2.0
        - horizontal[:, None]
        - azimuth_offsets[None, :]
    )

    phi = np.broadcast_to(
        elevations[None, :],
        theta.shape,
    )

    return (
        np.ascontiguousarray(theta.reshape(-1), dtype=np.float32),
        np.ascontiguousarray(phi.reshape(-1), dtype=np.float32),
    )


if __name__ == "__main__":
    theta, phi = generate_jt128()

    print("Number of rays:", len(theta))
    print("Expected:", 900 * 128)
    print("Theta range [deg]:", np.rad2deg(theta.min()), np.rad2deg(theta.max()))
    print("Phi range [deg]:", np.rad2deg(phi.min()), np.rad2deg(phi.max()))

    print("\nFirst 5 rays:")
    for i in range(5):
        print(
            f"theta={np.rad2deg(theta[i]):.3f} deg, "
            f"phi={np.rad2deg(phi[i]):.3f} deg"
        )

    print("\nFirst channel across first 5 horizontal positions:")
    for h in range(5):
        i = h * 128
        print(
            f"h={h}: "
            f"theta={np.rad2deg(theta[i]):.3f} deg, "
            f"phi={np.rad2deg(phi[i]):.3f} deg"
        )
