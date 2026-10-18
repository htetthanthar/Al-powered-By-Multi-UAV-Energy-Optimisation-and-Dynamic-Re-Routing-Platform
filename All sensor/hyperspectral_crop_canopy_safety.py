import math
import numpy as np
from typing import Dict, Tuple

class MultispectralCanopyAndSoilSafetyAnalyzer:
    def __init__(
        self,
        canopy_extinction_coeff_k: float = 0.65,
        min_bearing_capacity_kpa: float = 120.0, # 120 kPa required for 6kg drone landing gear
        waterlogging_ndwi_threshold: float = 0.25
    ):
        self.k_ext = canopy_extinction_coeff_k
        self.q_min_safe = min_bearing_capacity_kpa
        self.ndwi_water_thresh = waterlogging_ndwi_threshold

    def compute_vegetation_indices(
        self,
        reflectance_red: float,
        reflectance_red_edge: float,
        reflectance_nir: float,
        reflectance_green: float
    ) -> Dict[str, float]:
        """
        Calculates spectral bio-indices: NDVI, NDRE, NDWI.
        """
        r_red = max(reflectance_red, 1e-4)
        r_re = max(reflectance_red_edge, 1e-4)
        r_nir = max(reflectance_nir, 1e-4)
        r_green = max(reflectance_green, 1e-4)

        ndvi = (r_nir - r_red) / (r_nir + r_red)
        ndre = (r_nir - r_re) / (r_nir + r_re)
        # Normalized Difference Water Index: NDWI = (Green - NIR) / (Green + NIR)
        ndwi = (r_green - r_nir) / (r_green + r_nir)

        return {
            "ndvi": float(np.clip(ndvi, -1.0, 1.0)),
            "ndre": float(np.clip(ndre, -1.0, 1.0)),
            "ndwi": float(np.clip(ndwi, -1.0, 1.0))
        }

    def evaluate_delivery_surface_safety(
        self,
        multispectral_bands: Tuple[float, float, float, float], # (Red, RedEdge, NIR, Green)
        estimated_soil_moisture_fraction: float = 0.22
    ) -> Dict:
        """
        Evaluates canopy density, soil firmness, and waterlogging risk.
        """
        r_red, r_re, r_nir, r_green = multispectral_bands
        indices = self.compute_vegetation_indices(r_red, r_re, r_nir, r_green)

        ndvi = indices["ndvi"]
        ndwi = indices["ndwi"]

        # 1. Leaf Area Index (LAI) Canopy Model
        # LAI = -1/k * ln(1 - NDVI) for NDVI in [0, 0.95]
        clamped_ndvi = max(0.0, min(0.92, ndvi))
        lai = -(1.0 / self.k_ext) * math.log(1.0 - clamped_ndvi)
        canopy_height_m = lai * 0.45 # Empirical 0.45m height per LAI unit

        # 2. Semi-Empirical Soil Cone Index / Bearing Capacity (kPa)
        # Saturated muddy soil (high moisture) drops bearing capacity below 80 kPa
        moisture = np.clip(estimated_soil_moisture_fraction, 0.05, 0.65)
        # Dry hardpack: ~350 kPa, Saturated swamp: ~45 kPa
        q_cone_kpa = 420.0 * math.exp(-3.8 * moisture)

        # 3. Safety Clearance Rules
        is_waterlogged = ndwi >= self.ndwi_water_thresh or moisture >= 0.45
        is_canopy_too_dense = lai >= 3.5 # Dense orchard / maize prevents ground contact
        is_soil_firm = q_cone_kpa >= self.q_min_safe

        is_touchdown_safe = is_soil_firm and (not is_waterlogged) and (canopy_height_m <= 0.80)
        is_winch_drop_safe = (not is_waterlogged) and (canopy_height_m <= 1.50)

        action = "CLEAR_FOR_GEAR_TOUCHDOWN"
        if is_waterlogged:
            action = "ABORT_WATERLOGGED_MUD_HAZARD"
        elif is_canopy_too_dense:
            action = "ABORT_DENSE_CROP_CANOPY_SWITCH_TO_WINCH_DROP"
        elif not is_soil_firm:
            action = "SWITCH_TO_TETHERED_WINCH_HOVER_DROP"

        return {
            "ndvi_canopy_index": round(ndvi, 3),
            "ndre_chlorophyll_index": round(indices["ndre"], 3),
            "ndwi_water_index": round(ndwi, 3),
            "estimated_lai_density": round(lai, 2),
            "estimated_canopy_height_m": round(canopy_height_m, 2),
            "estimated_soil_bearing_capacity_kpa": round(q_cone_kpa, 1),
            "min_required_bearing_capacity_kpa": self.q_min_safe,
            "ground_touchdown_permitted": is_touchdown_safe,
            "winch_hover_drop_permitted": is_winch_drop_safe,
            "surface_safety_verdict": action
        }

if __name__ == "__main__":
    analyzer = MultispectralCanopyAndSoilSafetyAnalyzer()

    print("==========================================================================")
    print("      MULTISPECTRAL CROP CANOPY DENSITY & SOIL BEARING SAFETY AUDIT       ")
    print("==========================================================================")

    # 1. Firm Grass Field (Red=0.08, RedEdge=0.25, NIR=0.55, Green=0.12, Moisture=18%)
    res_grass = analyzer.evaluate_delivery_surface_safety(
        multispectral_bands=(0.08, 0.25, 0.55, 0.12),
        estimated_soil_moisture_fraction=0.18
    )
    print("Site #1 (Firm Grass Clearing):")
    for k, v in res_grass.items():
        print(f"  • {k:36s}: {v}")

    # 2. Waterlogged Rice Field / Mud (Red=0.15, RedEdge=0.22, NIR=0.32, Green=0.28, Moisture=52%)
    res_mud = analyzer.evaluate_delivery_surface_safety(
        multispectral_bands=(0.15, 0.22, 0.32, 0.28),
        estimated_soil_moisture_fraction=0.52
    )
    print("\nSite #2 (Waterlogged Saturated Soil):")
    for k, v in res_mud.items():
        print(f"  • {k:36s}: {v}")
    print("==========================================================================\n")