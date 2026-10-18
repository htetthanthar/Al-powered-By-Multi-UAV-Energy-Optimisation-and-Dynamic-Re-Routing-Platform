import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class OctreeNode:
    def __init__(self, center: np.ndarray, size: float, depth: int):
        self.center = center
        self.size = size
        self.depth = depth
        self.log_odds = 0.0 # Prior P = 0.5 (L = 0)
        self.children: Optional[List['OctreeNode']] = None

    def is_leaf(self) -> bool:
        return self.children is None

class OctoMapVoxelGridSlicer:
    def __init__(
        self,
        max_depth: int = 5,
        root_size_m: float = 64.0,
        resolution_m: float = 1.0,
        l_occ: float = 0.85, # Log-odds update on hit (~70% confidence)
        l_free: float = -0.40, # Log-odds update on miss (~40% confidence)
        occupancy_threshold: float = 0.50
    ):
        self.max_depth = max_depth
        self.root_size = root_size_m
        self.res = resolution_m
        self.l_occ = l_occ
        self.l_free = l_free
        self.p_thresh_log_odds = math.log(occupancy_threshold / (1.0 - occupancy_threshold))
        
        self.root = OctreeNode(center=np.array([0.0, 0.0, 0.0]), size=root_size_m, depth=0)

    def insert_point_cloud(self, sensor_origin: np.ndarray, point_cloud: np.ndarray):
        """
        Raycasts points from sensor origin to detected hits, updating log-odds along voxels.
        point_cloud shape: (N, 3)
        """
        for pt in point_cloud:
            self._update_ray_recursive(self.root, sensor_origin, pt, is_hit=True)

    def _update_ray_recursive(self, node: OctreeNode, origin: np.ndarray, end_point: np.ndarray, is_hit: bool):
        # Stop condition: reached max leaf depth or voxel resolution
        if node.depth >= self.max_depth or node.size <= self.res:
            node.log_odds += self.l_occ if is_hit else self.l_free
            # Clamping log-odds to prevent overconfidence [-3.5, 3.5]
            node.log_odds = np.clip(node.log_odds, -3.5, 3.5)
            return

        if node.is_leaf():
            # Subdivide into 8 octants
            node.children = []
            half = node.size / 4.0
            for dx in [-half, half]:
                for dy in [-half, half]:
                    for dz in [-half, half]:
                        new_center = node.center + np.array([dx, dy, dz])
                        node.children.append(OctreeNode(center=new_center, size=node.size / 2.0, depth=node.depth + 1))

        # Identify which child octant contains the endpoint
        for child in node.children:
            half_s = child.size / 2.0
            b_min = child.center - half_s
            b_max = child.center + half_s
            if np.all(end_point >= b_min) and np.all(end_point <= b_max):
                self._update_ray_recursive(child, origin, end_point, is_hit)
                break

    def extract_horizontal_clearance_slice(
        self,
        slice_altitude_z_m: float,
        extent_xy_m: float = 30.0,
        grid_step_m: float = 1.0
    ) -> Dict:
        """
        Evaluates a 2D horizontal plane slice at altitude Z to detect structural obstacles.
        """
        xs = np.arange(-extent_xy_m, extent_xy_m + 1e-6, grid_step_m)
        ys = np.arange(-extent_xy_m, extent_xy_m + 1e-6, grid_step_m)
        grid_occupancy = np.zeros((len(ys), len(xs)))

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                query_pt = np.array([x, y, slice_altitude_z_m])
                occ_prob = self._query_point_occupancy(self.root, query_pt)
                grid_occupancy[i, j] = occ_prob

        # Evaluate circular drop clearance corridor (radius = 5.0m)
        xx, yy = np.meshgrid(xs, ys)
        center_mask = (xx**2 + yy**2) <= (5.0 ** 2)
        max_corridor_occupancy = float(np.max(grid_occupancy[center_mask]))
        corridor_clear = max_corridor_occupancy < 0.40

        return {
            "slice_altitude_z_m": round(slice_altitude_z_m, 2),
            "queried_grid_cells_count": grid_occupancy.size,
            "max_occupancy_probability_in_drop_corridor": round(max_corridor_occupancy, 3),
            "delivery_corridor_clear": corridor_clear,
            "action": "PROCEED_WITH_WINCH_DESCENT" if corridor_clear else "ABORT_ELEVATION_OBSTACLE_DETECTED"
        }

    def _query_point_occupancy(self, node: OctreeNode, point: np.ndarray) -> float:
        if node.is_leaf():
            log_odds = node.log_odds
            prob = 1.0 / (1.0 + math.exp(-log_odds))
            return prob

        for child in node.children:
            half_s = child.size / 2.0
            if np.all(point >= child.center - half_s) and np.all(point <= child.center + half_s):
                return self._query_point_occupancy(child, point)

        return 0.5 # Unknown prior

if __name__ == "__main__":
    slicer = OctoMapVoxelGridSlicer(max_depth=5, root_size_m=64.0, resolution_m=1.0)

    # Simulate point cloud from building wall at x in [8, 12], y in [-10, 10], z in [0, 30]
    np.random.seed(42)
    n_pts = 350
    wall_x = np.random.uniform(8.0, 12.0, n_pts)
    wall_y = np.random.uniform(-10.0, 10.0, n_pts)
    wall_z = np.random.uniform(0.0, 30.0, n_pts)
    cloud = np.column_stack([wall_x, wall_y, wall_z])

    sensor_pos = np.array([0.0, 0.0, 25.0])
    slicer.insert_point_cloud(sensor_pos, cloud)

    slice_audit = slicer.extract_horizontal_clearance_slice(slice_altitude_z_m=25.0, extent_xy_m=20.0)

    print("==========================================================================")
    print("      LIDAR 3D OCTOMAP OCCUPANCY & HORIZONTAL CLEARANCE SLICER AUDIT      ")
    print("==========================================================================")
    for k, v in slice_audit.items():
        print(f"  • {k:45s}: {v}")
    print("==========================================================================\n")