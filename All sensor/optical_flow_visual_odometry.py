import math
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List

class MonocularVisualOdometryDeadReckoning:
    def __init__(
        self,
        focal_length_px: float = 520.0,
        optical_center: Tuple[float, float] = (320.0, 240.0),
        max_features_to_track: int = 200,
        min_feature_distance_px: float = 15.0
    ):
        self.fx = focal_length_px
        self.fy = focal_length_px
        self.cx, self.cy = optical_center
        self.max_features = max_features_to_track
        self.min_dist = min_feature_distance_px

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        
        # Dead-reckoning state accumulation in local navigation frame
        self.integrated_x_m = 0.0
        self.integrated_y_m = 0.0
        self.integrated_yaw_rad = 0.0

    def process_camera_frame(
        self,
        current_frame_bgr: np.ndarray,
        rangefinder_altitude_agl_m: float,
        dt_seconds: float
    ) -> Dict:
        """
        Calculates horizontal metric translation (Delta_X, Delta_Y) scaled by true AGL altitude.
        """
        curr_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
        h_agl = max(rangefinder_altitude_agl_m, 0.5)

        delta_x_m = 0.0
        delta_y_m = 0.0
        delta_yaw_rad = 0.0
        tracked_feature_count = 0

        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 10:
            # 1. Pyramidal Lucas-Kanade Sparse Feature Tracking
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray,
                curr_gray,
                self.prev_pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Filter valid tracked correspondences
            valid_prev = self.prev_pts[status.flatten() == 1]
            valid_next = next_pts[status.flatten() == 1]
            tracked_feature_count = len(valid_next)

            if tracked_feature_count >= 8:
                # 2. Estimate 2D Homography / Rigid Affine Transform with RANSAC
                transform_matrix, inliers = cv2.estimateAffinePartial2D(
                    valid_prev,
                    valid_next,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0
                )

                if transform_matrix is not None:
                    # Extract pixel translation and rotation
                    dx_px = transform_matrix[0, 2]
                    dy_px = transform_matrix[1, 2]
                    delta_yaw_rad = -math.atan2(transform_matrix[1, 0], transform_matrix[0, 0])

                    # 3. Metric Scaling via Pinhole Model: Delta_X_metric = (dx_px * Z) / f
                    # In NED camera frame: camera +X is Right (East), +Y is Down (South)
                    delta_x_m = (dx_px * h_agl) / self.fx
                    delta_y_m = (dy_px * h_agl) / self.fy

                    # Rotate into current dead-reckoned world frame
                    cos_yaw = math.cos(self.integrated_yaw_rad)
                    sin_yaw = math.sin(self.integrated_yaw_rad)
                    
                    world_dx = delta_x_m * cos_yaw - delta_y_m * sin_yaw
                    world_dy = delta_x_m * sin_yaw + delta_y_m * cos_yaw

                    self.integrated_x_m += world_dx
                    self.integrated_y_m += world_dy
                    self.integrated_yaw_rad += delta_yaw_rad

        # 4. Feature Replenishment
        if self.prev_pts is None or tracked_feature_count < 50:
            new_corners = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=self.max_features,
                qualityLevel=0.01,
                minDistance=self.min_dist,
                blockSize=3
            )
            self.prev_pts = new_corners
        else:
            self.prev_pts = valid_next.reshape(-1, 1, 2)

        self.prev_gray = curr_gray

        speed_ms = math.hypot(delta_x_m, delta_y_m) / max(dt_seconds, 1e-4)

        return {
            "dead_reckoning_x_m": round(self.integrated_x_m, 2),
            "dead_reckoning_y_m": round(self.integrated_y_m, 2),
            "estimated_yaw_deg": round(math.degrees(self.integrated_yaw_rad), 1),
            "instantaneous_speed_ms": round(speed_ms, 2),
            "tracked_features_inliers": tracked_feature_count,
            "vio_confidence": "HIGH" if tracked_feature_count >= 40 else ("DEGRADED" if tracked_feature_count >= 10 else "LOST")
        }

if __name__ == "__main__":
    vio = MonocularVisualOdometryDeadReckoning()

    # Generate synthetic downward ground texture sequence
    np.random.seed(42)
    base_texture = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    print("==========================================================================")
    print("      GPS-DENIED VISUAL-INERTIAL ODOMETRY (VIO) DEAD RECKONING TEST       ")
    print("==========================================================================")

    # Simulate drone translating forward by 0.9 meters per frame at 25m altitude
    for frame_idx in range(1, 6):
        shift_px = int(frame_idx * 18.7) # 18.7px shift ~ 0.9m at 25m AGL
        shifted_frame = np.roll(base_texture, shift_px, axis=1)

        result = vio.process_camera_frame(
            current_frame_bgr=shifted_frame,
            rangefinder_altitude_agl_m=25.0,
            dt_seconds=0.05
        )
        print(f"Frame #{frame_idx:02d}: Est Pos = [X: {result['dead_reckoning_x_m']:5.2f}m, Y: {result['dead_reckoning_y_m']:5.2f}m] | "
              f"Speed: {result['instantaneous_speed_ms']:5.2f} m/s | Features: {result['tracked_features_inliers']:03d} | Status: {result['vio_confidence']}")
    print("==========================================================================\n")