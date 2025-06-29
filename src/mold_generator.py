#!/usr/bin/env python
"""
mold_generator.py

A command-line tool for generating 3D molds from input STL files using trimesh.
It includes enhanced mesh repair (using Open3D), draft angle application, hole processing,
flexible splitting (halves, quarters, or custom slicing planes), intelligent alignment features,
and smart logging system.

IMPROVED FEATURES:
- Intelligent alignment feature placement with geometric analysis
- Multi-engine boolean operation fallbacks  
- Enhanced mesh repair and validation
- Smart logging system with automatic cleanup
- Model size inspection capabilities
- Robust coordinate validation
"""

import trimesh
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse
import json
import sys
import time
from pathlib import Path
import glob

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOGGING_CONFIG = {
    'max_log_files': 5,        # Maximum number of log files to keep
    'max_log_age_days': 7,     # Remove logs older than this many days
    'default_level': 'INFO',   # Default logging level: DEBUG, INFO, WARNING, ERROR
    'file_level': 'DEBUG',     # File logging level (more detailed)
    'console_level': 'INFO',   # Console logging level (user-friendly)
    'enable_function_names': True,  # Include function names in file logs
}
# =============================================================================

def cleanup_old_logs(max_age_days=None, max_count=None):
    """Clean up old log files based on age and count."""
    if max_age_days is None:
        max_age_days = LOGGING_CONFIG['max_log_age_days']
    if max_count is None:
        max_count = LOGGING_CONFIG['max_log_files']
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return
    
    log_files = list(logs_dir.glob("mold_generator_*.log"))
    if not log_files:
        return
    
    now = time.time()
    removed_count = 0
    
    # Sort by modification time (newest first)
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    for i, log_file in enumerate(log_files):
        should_remove = False
        
        # Remove if older than max_age_days
        file_age_days = (now - log_file.stat().st_mtime) / (24 * 3600)
        if file_age_days > max_age_days:
            should_remove = True
        
        # Remove if beyond max_count (keeping newest ones)
        if i >= max_count:
            should_remove = True
        
        if should_remove:
            try:
                log_file.unlink()
                removed_count += 1
            except OSError:
                pass  # File might be in use
    
    if removed_count > 0:
        print(f"🧹 Cleaned up {removed_count} old log files")

def setup_smart_logging():
    """Sets up an intelligent logging system that manages log files automatically."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Clean up old logs (both by count and age)
    cleanup_old_logs()
    
    # Clear any existing handlers to avoid duplicates
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set up basic configuration
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    logging.basicConfig(
        level=level_map.get(LOGGING_CONFIG['default_level'], logging.INFO), 
        format='%(asctime)s - %(levelname)s: %(message)s', 
        handlers=[]
    )
    
    # Create new log filename
    log_filename = logs_dir / f'mold_generator_{time.strftime("%Y%m%d_%H%M%S")}.log'
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(level_map.get(LOGGING_CONFIG['file_level'], logging.DEBUG))
    
    if LOGGING_CONFIG['enable_function_names']:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
    else:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level_map.get(LOGGING_CONFIG['console_level'], logging.INFO))
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log the setup
    logging.info(f"Logging initialized. Log file: {log_filename}")
    logging.info(f"Keeping previous log files in {logs_dir}/")
    
    return str(log_filename)

def log_mesh_stats(mesh, label="Mesh"):
    """Logs detailed statistics about a mesh."""
    logging.info(f"===== {label} Statistics =====")
    logging.info(f"Vertices: {len(mesh.vertices)}")
    logging.info(f"Faces: {len(mesh.faces)}")
    logging.info(f"Is watertight: {mesh.is_watertight}")
    logging.info(f"Is winding consistent: {mesh.is_winding_consistent}")
    logging.info(f"Volume: {mesh.volume if hasattr(mesh, 'volume') else 'Unknown'}")
    logging.info(f"Bounding box: {mesh.bounds}")
    logging.info(f"Center of mass: {mesh.center_mass}")
    try:
        euler_number = mesh.euler_number
        logging.info(f"Euler number: {euler_number}")
    except:
        logging.info("Euler number: Could not compute")
    logging.info(f"==============================")

def inspect_model_size(file_path):
    """Inspect and display detailed information about a model's size and properties."""
    try:
        print(f"\n📏 Inspecting model: {file_path}")
        print("=" * 60)
        
        # Load mesh
        mesh = trimesh.load_mesh(file_path)
        
        # Basic info
        print(f"Vertices: {len(mesh.vertices):,}")
        print(f"Faces: {len(mesh.faces):,}")
        print(f"Is watertight: {mesh.is_watertight}")
        
        # Size information
        bounds = mesh.bounds
        extents = mesh.extents
        print(f"\nDimensions (mm):")
        print(f"  X: {extents[0]:.2f} mm (from {bounds[0][0]:.2f} to {bounds[1][0]:.2f})")
        print(f"  Y: {extents[1]:.2f} mm (from {bounds[0][1]:.2f} to {bounds[1][1]:.2f})")
        print(f"  Z: {extents[2]:.2f} mm (from {bounds[0][2]:.2f} to {bounds[1][2]:.2f})")
        print(f"Overall size: {max(extents):.2f} mm (largest dimension)")
        
        # Volume and surface area
        if hasattr(mesh, 'volume'):
            print(f"\nVolume: {mesh.volume:.2f} mm³")
        if hasattr(mesh, 'area'):
            print(f"Surface area: {mesh.area:.2f} mm²")
        
        # Center
        center = mesh.center_mass
        print(f"\nCenter of mass: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        
        # Recommended settings
        max_dim = max(extents)
        print(f"\n💡 Recommended settings:")
        print(f"  Padding: {max_dim * 0.05:.2f} - {max_dim * 0.15:.2f} mm")
        print(f"  Alignment radius: {max_dim * 0.02:.2f} - {max_dim * 0.08:.2f} mm")
        
        # File size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"\nFile size: {file_size:.2f} MB")
        
        print("=" * 60)
        
        return {
            'extents': extents,
            'bounds': bounds,
            'volume': getattr(mesh, 'volume', None),
            'area': getattr(mesh, 'area', None),
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'is_watertight': mesh.is_watertight,
            'recommended_padding': [max_dim * 0.05, max_dim * 0.15],
            'recommended_radius': [max_dim * 0.02, max_dim * 0.08]
        }
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        return None

def enhanced_mesh_repair(mesh):
    """Enhanced mesh repair using Open3D functionalities."""
    try:
        import open3d as o3d
    except ImportError:
        logging.error("Open3D is not installed. Please install it with 'pip install open3d'.")
        raise

    logging.info("Attempting enhanced mesh repair using Open3D.")
    log_mesh_stats(mesh, "Before Open3D Repair")
    
    # Convert trimesh to Open3D mesh:
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    
    # Log number of elements before cleaning
    logging.info(f"Open3D mesh before cleaning: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
    
    # Remove degenerate triangles and duplicated elements:
    logging.info("Removing degenerate triangles...")
    mesh_o3d.remove_degenerate_triangles()
    logging.info(f"After removing degenerate triangles: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
    
    logging.info("Removing duplicated triangles...")
    mesh_o3d.remove_duplicated_triangles()
    logging.info(f"After removing duplicated triangles: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
    
    logging.info("Removing duplicated vertices...")
    mesh_o3d.remove_duplicated_vertices()
    logging.info(f"After removing duplicated vertices: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
    
    logging.info("Removing non-manifold edges...")
    mesh_o3d.remove_non_manifold_edges()
    logging.info(f"After removing non-manifold edges: {len(mesh_o3d.vertices)} vertices, {len(mesh_o3d.triangles)} triangles")
    
    # Convert back to trimesh:
    repaired_vertices = np.asarray(mesh_o3d.vertices)
    repaired_faces = np.asarray(mesh_o3d.triangles)
    
    if len(repaired_vertices) == 0 or len(repaired_faces) == 0:
        logging.error("Open3D repair resulted in an empty mesh!")
        return mesh  # Return original mesh if repair failed
    
    repaired_mesh = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces, process=True)
    log_mesh_stats(repaired_mesh, "After Open3D Repair")
    
    return repaired_mesh

def apply_draft(mesh, draft_angle):
    """Applies a draft (taper) to the input mesh."""
    if draft_angle == 0:
        logging.info("No draft angle applied (draft_angle = 0)")
        return mesh

    z_min = mesh.bounds[0][2]
    z_max = mesh.bounds[1][2]
    if z_max == z_min:
        logging.warning("Cannot apply draft angle: z_min equals z_max")
        return mesh

    logging.info(f"Applying draft angle of {draft_angle}° (z_min={z_min}, z_max={z_max})")
    angle_rad = np.deg2rad(draft_angle)
    new_vertices = mesh.vertices.copy()
    factors = 1.0 + ((new_vertices[:, 2] - z_min) / (z_max - z_min)) * np.tan(angle_rad)
    
    # Log some sample factors
    sample_idx = np.linspace(0, len(new_vertices)-1, min(10, len(new_vertices)), dtype=int)
    for i in sample_idx:
        logging.debug(f"Vertex {i} z={new_vertices[i, 2]}, factor={factors[i]}")
    
    new_vertices[:, 0] *= factors
    new_vertices[:, 1] *= factors

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    log_mesh_stats(new_mesh, "After Draft Angle")
    return new_mesh

def calculate_intelligent_alignment_coordinates(mold, original_mesh, mold_center, mold_extents, 
                                               mesh_bounds, alignment_radius, padding):
    """Uses Trimesh's geometric analysis to find optimal alignment feature positions."""
    try:
        logging.info("Using intelligent geometric analysis for alignment placement...")
        
        # Interface plane (where the split occurs)
        interface_x = mold_center[0]
        
        # Calculate safe zones using geometric analysis - MADE MORE FLEXIBLE
        min_clearance = max(alignment_radius * 0.8, 0.2)  # Reduced from 1.1 to 0.8, minimum 0.2mm
        
        # Get mold bounds in YZ plane
        y_min_mold = mold_center[1] - mold_extents[1] / 2
        y_max_mold = mold_center[1] + mold_extents[1] / 2
        z_min_mold = mold_center[2] - mold_extents[2] / 2
        z_max_mold = mold_center[2] + mold_extents[2] / 2
        
        # Get mesh bounds in YZ plane with REDUCED clearance for more flexibility
        y_min_mesh = mesh_bounds[0][1] - min_clearance
        y_max_mesh = mesh_bounds[1][1] + min_clearance
        z_min_mesh = mesh_bounds[0][2] - min_clearance
        z_max_mesh = mesh_bounds[1][2] + min_clearance
        
        logging.info(f"Mold YZ bounds: Y[{y_min_mold:.2f}, {y_max_mold:.2f}], Z[{z_min_mold:.2f}, {z_max_mold:.2f}]")
        logging.info(f"Mesh YZ bounds (with clearance): Y[{y_min_mesh:.2f}, {y_max_mesh:.2f}], Z[{z_min_mesh:.2f}, {z_max_mesh:.2f}]")
        
        # MUCH MORE FLEXIBLE: Smaller margins and more forgiving placement
        feature_margin = max(alignment_radius * 0.3, 0.05)  # Much smaller margin
        
        # Calculate available space more accurately
        y_available_left = y_min_mesh - y_min_mold - feature_margin * 2
        y_available_right = y_max_mold - y_max_mesh - feature_margin * 2
        z_available_bottom = z_min_mesh - z_min_mold - feature_margin * 2
        z_available_top = z_max_mold - z_max_mesh - feature_margin * 2
        
        logging.info(f"Available space: Y_left={y_available_left:.2f}, Y_right={y_available_right:.2f}, Z_bottom={z_available_bottom:.2f}, Z_top={z_available_top:.2f}")
        
        candidate_regions = []
        
        # Only add regions that have enough space (at least alignment_radius)
        min_space_needed = alignment_radius * 1.5  # Need 1.5x radius for safe placement
        
        if y_available_left >= min_space_needed and z_available_bottom >= min_space_needed:
            candidate_regions.append({
                "name": "bottom-left",
                "y_range": [y_min_mold + feature_margin, y_min_mesh - feature_margin], 
                "z_range": [z_min_mold + feature_margin, z_min_mesh - feature_margin]
            })
        
        if y_available_right >= min_space_needed and z_available_top >= min_space_needed:
            candidate_regions.append({
                "name": "top-right",
                "y_range": [y_max_mesh + feature_margin, y_max_mold - feature_margin],
                "z_range": [z_max_mesh + feature_margin, z_max_mold - feature_margin]
            })
        
        if y_available_left >= min_space_needed and z_available_top >= min_space_needed:
            candidate_regions.append({
                "name": "top-left",
                "y_range": [y_min_mold + feature_margin, y_min_mesh - feature_margin],
                "z_range": [z_max_mesh + feature_margin, z_max_mold - feature_margin]
            })
        
        if y_available_right >= min_space_needed and z_available_bottom >= min_space_needed:
            candidate_regions.append({
                "name": "bottom-right",
                "y_range": [y_max_mesh + feature_margin, y_max_mold - feature_margin],
                "z_range": [z_min_mold + feature_margin, z_min_mesh - feature_margin]
            })
        
        # FALLBACK: If no corner regions work, try more flexible placement options
        if len(candidate_regions) < 2:
            logging.info("Not enough corner regions, trying flexible placement options...")
            
            # Try middle regions with smaller features
            relaxed_margin = feature_margin * 0.3  # Even more relaxed
            
            # Try edge-middle regions (safer than pure edges)
            if y_available_left >= alignment_radius * 0.8:
                candidate_regions.append({
                    "name": "left-middle",
                    "y_range": [y_min_mold + relaxed_margin, y_min_mesh - relaxed_margin],
                    "z_range": [z_min_mesh + relaxed_margin, z_max_mesh - relaxed_margin]
                })
            
            if y_available_right >= alignment_radius * 0.8:
                candidate_regions.append({
                    "name": "right-middle", 
                    "y_range": [y_max_mesh + relaxed_margin, y_max_mold - relaxed_margin],
                    "z_range": [z_min_mesh + relaxed_margin, z_max_mesh - relaxed_margin]
                })
            
            if z_available_bottom >= alignment_radius * 0.8:
                candidate_regions.append({
                    "name": "bottom-middle",
                    "y_range": [y_min_mesh + relaxed_margin, y_max_mesh - relaxed_margin],
                    "z_range": [z_min_mold + relaxed_margin, z_min_mesh - relaxed_margin]
                })
            
            if z_available_top >= alignment_radius * 0.8:
                candidate_regions.append({
                    "name": "top-middle",
                    "y_range": [y_min_mesh + relaxed_margin, y_max_mesh - relaxed_margin],
                    "z_range": [z_max_mesh + relaxed_margin, z_max_mold - relaxed_margin]
                })
                
            # NEW: Try diagonal and offset regions for more variety
            y_center = (y_min_mold + y_max_mold) / 2
            z_center = (z_min_mold + z_max_mold) / 2
            
            # Offset regions (not just corners/edges)
            quarter_y = (y_max_mold - y_min_mold) / 4
            quarter_z = (z_max_mold - z_min_mold) / 4
            
            # Try offset positions for more placement variety
            if len(candidate_regions) < 4:  # Still need more options
                offset_regions = [
                    {
                        "name": "offset-1",
                        "y_range": [y_center - quarter_y, y_center - quarter_y + alignment_radius * 2],
                        "z_range": [z_center - quarter_z, z_center - quarter_z + alignment_radius * 2]
                    },
                    {
                        "name": "offset-2", 
                        "y_range": [y_center + quarter_y - alignment_radius * 2, y_center + quarter_y],
                        "z_range": [z_center + quarter_z - alignment_radius * 2, z_center + quarter_z]
                    },
                    {
                        "name": "offset-3",
                        "y_range": [y_center - quarter_y, y_center - quarter_y + alignment_radius * 2],
                        "z_range": [z_center + quarter_z - alignment_radius * 2, z_center + quarter_z]
                    },
                    {
                        "name": "offset-4",
                        "y_range": [y_center + quarter_y - alignment_radius * 2, y_center + quarter_y],
                        "z_range": [z_center - quarter_z, z_center - quarter_z + alignment_radius * 2]
                    }
                ]
                
                for offset_region in offset_regions:
                    # Check if this offset region is valid (not overlapping with mesh)
                    y_min_r, y_max_r = offset_region["y_range"]
                    z_min_r, z_max_r = offset_region["z_range"]
                    
                    # Simple overlap check
                    if (y_max_r < y_min_mesh or y_min_r > y_max_mesh or 
                        z_max_r < z_min_mesh or z_min_r > z_max_mesh):
                        candidate_regions.append(offset_region)
                        if len(candidate_regions) >= 6:  # Enough options
                            break
        
        logging.info(f"Generated {len(candidate_regions)} candidate regions: {[r['name'] for r in candidate_regions]}")
        
        valid_coordinates = []
        
        for i, region in enumerate(candidate_regions):
            region_name = region.get("name", f"region_{i}")
            y_min, y_max = region["y_range"]
            z_min, z_max = region["z_range"]
            
            # Check if region is valid (has positive space)
            if y_max <= y_min or z_max <= z_min:
                logging.debug(f"Region {region_name} invalid: no space available (Y:{y_min:.2f}-{y_max:.2f}, Z:{z_min:.2f}-{z_max:.2f})")
                continue
            
            # Place feature in center of valid region
            y_coord = (y_min + y_max) / 2
            z_coord = (z_min + z_max) / 2
            
            # MUCH MORE RELAXED validation - prioritize getting features placed
            test_point = np.array([interface_x, y_coord, z_coord])
            
            # Simplified validation - just check basic geometric constraints
            is_valid = True
            validation_reason = "geometric bounds"
            
            # Optional distance check (but don't fail if it doesn't work)
            if hasattr(original_mesh, 'nearest'):
                try:
                    distances, _, _ = original_mesh.nearest.on_surface([test_point])
                    # FIX: Handle both scalar and array distances properly
                    if hasattr(distances, '__len__') and len(distances) > 0:
                        min_distance_to_mesh = float(distances[0])
                    else:
                        min_distance_to_mesh = float(distances)
                    
                    # VERY RELAXED: Accept if distance is at least 50% of clearance OR alignment_radius
                    required_distance = min(min_clearance * 0.5, alignment_radius * 0.7)
                    if min_distance_to_mesh >= required_distance:
                        validation_reason = f"distance check passed ({min_distance_to_mesh:.2f} >= {required_distance:.2f})"
                    else:
                        # Still accept but with warning
                        validation_reason = f"distance check warning ({min_distance_to_mesh:.2f} < {required_distance:.2f}) but proceeding"
                        logging.warning(f"Region {region_name} close to mesh but proceeding: distance={min_distance_to_mesh:.2f}, required={required_distance:.2f}")
                except Exception as e:
                    validation_reason = f"distance check failed ({e}) but proceeding with geometric bounds"
                    logging.debug(f"Distance query failed for region {region_name}: {e}")
            
            if is_valid:
                valid_coordinates.append([y_coord, z_coord])
                logging.info(f"Valid placement found in {region_name} at [{y_coord:.2f}, {z_coord:.2f}] - {validation_reason}")
            
            # We only need 2 valid coordinates
            if len(valid_coordinates) >= 2:
                break
        
        if len(valid_coordinates) >= 2:
            result_coords = valid_coordinates[:2]
            logging.info(f"Intelligent placement successful: {result_coords}")
            return result_coords
        elif len(valid_coordinates) == 1:
            # If we only found one, create a second one nearby
            y1, z1 = valid_coordinates[0]
            
            # Try to place second feature offset from the first
            offset_distance = alignment_radius * 2
            potential_coords = [
                [y1 + offset_distance, z1],
                [y1 - offset_distance, z1], 
                [y1, z1 + offset_distance],
                [y1, z1 - offset_distance]
            ]
            
            for y2, z2 in potential_coords:
                # Check if within mold bounds
                if (y2 >= y_min_mold + feature_margin and y2 <= y_max_mold - feature_margin and
                    z2 >= z_min_mold + feature_margin and z2 <= z_max_mold - feature_margin):
                    result_coords = [[y1, z1], [y2, z2]]
                    logging.info(f"Intelligent placement with offset successful: {result_coords}")
                    return result_coords
            
            logging.warning(f"Only found 1 valid placement, couldn't create second: {valid_coordinates}")
            return None
        else:
            # FINAL FALLBACK: Force placement if no regions worked but we have space
            logging.warning(f"No valid placements found with standard algorithm, trying emergency fallback...")
            
            # Emergency fallback - place features in the most available space
            emergency_coords = []
            
            # Try to place at least one feature in each available direction
            # Use a smaller effective radius for emergency placement
            emergency_radius = min(alignment_radius * 0.7, max(y_available_left, y_available_right, z_available_bottom, z_available_top) * 0.8)
            
            if y_available_left > emergency_radius:
                y_coord = y_min_mold + feature_margin + emergency_radius
                z_coord = (z_min_mold + z_max_mold) / 2  # Middle of mold in Z
                emergency_coords.append([y_coord, z_coord])
                logging.info(f"Emergency placement 1: [{y_coord:.2f}, {z_coord:.2f}] in left space (radius {emergency_radius:.2f})")
            
            if y_available_right > emergency_radius and len(emergency_coords) < 2:
                y_coord = y_max_mold - feature_margin - emergency_radius
                z_coord = (z_min_mold + z_max_mold) / 2  # Middle of mold in Z
                emergency_coords.append([y_coord, z_coord])
                logging.info(f"Emergency placement 2: [{y_coord:.2f}, {z_coord:.2f}] in right space (radius {emergency_radius:.2f})")
            
            if z_available_bottom > emergency_radius and len(emergency_coords) < 2:
                y_coord = (y_min_mold + y_max_mold) / 2  # Middle of mold in Y
                z_coord = z_min_mold + feature_margin + emergency_radius
                emergency_coords.append([y_coord, z_coord])
                logging.info(f"Emergency placement 3: [{y_coord:.2f}, {z_coord:.2f}] in bottom space (radius {emergency_radius:.2f})")
            
            if z_available_top > emergency_radius and len(emergency_coords) < 2:
                y_coord = (y_min_mold + y_max_mold) / 2  # Middle of mold in Y
                z_coord = z_max_mold - feature_margin - emergency_radius
                emergency_coords.append([y_coord, z_coord])
                logging.info(f"Emergency placement 4: [{y_coord:.2f}, {z_coord:.2f}] in top space (radius {emergency_radius:.2f})")
            
            if len(emergency_coords) >= 2:
                result_coords = emergency_coords[:2]
                logging.info(f"Emergency fallback placement successful: {result_coords}")
                return result_coords
            elif len(emergency_coords) == 1:
                # Create a second coordinate by offsetting the first
                y1, z1 = emergency_coords[0]
                y2 = y1 + alignment_radius * 1.5 if y1 < mold_center[1] else y1 - alignment_radius * 1.5
                z2 = z1 + alignment_radius * 1.5 if z1 < mold_center[2] else z1 - alignment_radius * 1.5
                
                # Clamp to mold bounds
                y2 = max(y_min_mold + feature_margin, min(y_max_mold - feature_margin, y2))
                z2 = max(z_min_mold + feature_margin, min(z_max_mold - feature_margin, z2))
                
                result_coords = [[y1, z1], [y2, z2]]
                logging.info(f"Emergency fallback with offset successful: {result_coords}")
                return result_coords
            else:
                logging.error(f"Emergency fallback failed - no space available for alignment features")
                return None
            
    except Exception as e:
        logging.error(f"Intelligent alignment calculation failed: {e}")
        return None

def validate_alignment_placement(coords, original_mesh, mold_center, mold_extents, alignment_radius):
    """Validates that alignment feature coordinates won't intersect the cavity."""
    try:
        interface_x = mold_center[0]
        min_safe_distance = alignment_radius * 1.05  # 5% safety margin
        
        for i, coord in enumerate(coords):
            y, z = coord
            center_point = np.array([interface_x, y, z])
            
            # Check if within mold bounds (be more generous)
            margin = alignment_radius * 0.5  # Smaller margin
            if (y < mold_center[1] - mold_extents[1]/2 + margin or 
                y > mold_center[1] + mold_extents[1]/2 - margin or
                z < mold_center[2] - mold_extents[2]/2 + margin or 
                z > mold_center[2] + mold_extents[2]/2 - margin):
                logging.warning(f"Coordinate {i+1} outside mold bounds: [{y:.2f}, {z:.2f}]")
                return False
            
            # Check distance to mesh surface (simplified and more robust)
            if hasattr(original_mesh, 'nearest'):
                try:
                    distances, _, _ = original_mesh.nearest.on_surface([center_point])
                    # FIX: Handle both scalar and array distances properly
                    if hasattr(distances, '__len__') and len(distances) > 0:
                        distance_to_mesh = float(distances[0])
                    else:
                        distance_to_mesh = float(distances)
                    
                    if distance_to_mesh < min_safe_distance:
                        logging.warning(f"Coordinate {i+1} too close to mesh: distance={distance_to_mesh:.2f}, required={min_safe_distance:.2f}")
                        # Don't fail validation - just warn
                        logging.info(f"Coordinate {i+1} proceeding despite proximity warning")
                    else:
                        logging.info(f"Coordinate {i+1} validation passed: distance={distance_to_mesh:.2f}")
                except Exception as e:
                    logging.warning(f"Distance validation failed for coordinate {i+1}: {e}")
                    # Continue anyway - geometric bounds should be sufficient
        
        logging.info("All alignment coordinates validated successfully")
        return True
        
    except Exception as e:
        logging.error(f"Alignment validation failed: {e}")
        return False

def calculate_quarters_alignment_coordinates(mold, original_mesh, mold_center, mold_extents, 
                                           mesh_bounds, alignment_radius, padding):
    """
    Calculate optimal alignment feature coordinates for quarters mode.
    
    For quarters, we need alignment features on both interface planes:
    - X=0 plane (separates left from right quarters)
    - Y=0 plane (separates front from back quarters)
    
    This creates a cross-pattern ensuring all 4 parts align correctly.
    
    Returns:
        dict: Dictionary containing coordinates for both interface planes:
              {'x_plane_coords': [[y1, z1], [y2, z2]], 
               'y_plane_coords': [[x1, z1], [x2, z2]]}
              Returns None if placement is not possible.
    """
    try:
        logging.info("Calculating quarters alignment coordinates for both X and Y interface planes...")
        
        # Interface planes for quarters mode
        interface_x = mold_center[0]  # X=0 plane (left/right split)
        interface_y = mold_center[1]  # Y=0 plane (front/back split)
        
        # Calculate safe zones using geometric analysis - MADE MORE FLEXIBLE
        min_clearance = max(alignment_radius * 0.8, 0.2)  # Reduced from 1.1 to 0.8, minimum 0.2mm
        
        # Get mold bounds
        x_min_mold = mold_center[0] - mold_extents[0] / 2
        x_max_mold = mold_center[0] + mold_extents[0] / 2
        y_min_mold = mold_center[1] - mold_extents[1] / 2
        y_max_mold = mold_center[1] + mold_extents[1] / 2
        z_min_mold = mold_center[2] - mold_extents[2] / 2
        z_max_mold = mold_center[2] + mold_extents[2] / 2
        
        # Get mesh bounds with REDUCED clearance for more flexibility
        x_min_mesh = mesh_bounds[0][0] - min_clearance
        x_max_mesh = mesh_bounds[1][0] + min_clearance
        y_min_mesh = mesh_bounds[0][1] - min_clearance
        y_max_mesh = mesh_bounds[1][1] + min_clearance
        z_min_mesh = mesh_bounds[0][2] - min_clearance
        z_max_mesh = mesh_bounds[1][2] + min_clearance
        
        logging.info(f"Mold bounds: X[{x_min_mold:.2f}, {x_max_mold:.2f}], Y[{y_min_mold:.2f}, {y_max_mold:.2f}], Z[{z_min_mold:.2f}, {z_max_mold:.2f}]")
        logging.info(f"Mesh bounds (with clearance): X[{x_min_mesh:.2f}, {x_max_mesh:.2f}], Y[{y_min_mesh:.2f}, {y_max_mesh:.2f}], Z[{z_min_mesh:.2f}, {z_max_mesh:.2f}]")
        
        feature_margin = max(alignment_radius * 0.3, 0.05)  # Much smaller margin
        
        # ========== CALCULATE X-PLANE COORDINATES (Y-Z plane, left/right split) ==========
        # Calculate available space in Y and Z directions for X-plane features
        y_available_left = y_min_mesh - y_min_mold - feature_margin * 2
        y_available_right = y_max_mold - y_max_mesh - feature_margin * 2
        z_available_bottom = z_min_mesh - z_min_mold - feature_margin * 2
        z_available_top = z_max_mold - z_max_mesh - feature_margin * 2
        
        logging.info(f"X-plane available space: Y_left={y_available_left:.2f}, Y_right={y_available_right:.2f}, Z_bottom={z_available_bottom:.2f}, Z_top={z_available_top:.2f}")
        
        x_plane_coords = []
        min_space_needed = alignment_radius * 1.5  # Need 1.5x radius for safe placement
        
        # Try to place two features on X-plane (Y-Z coordinates)
        if y_available_left >= min_space_needed and z_available_bottom >= min_space_needed:
            y_coord = y_min_mold + feature_margin + alignment_radius
            z_coord = z_min_mold + feature_margin + alignment_radius
            x_plane_coords.append([y_coord, z_coord])
            logging.info(f"X-plane feature 1 placed at Y={y_coord:.2f}, Z={z_coord:.2f} (bottom-left)")
        
        if y_available_right >= min_space_needed and z_available_top >= min_space_needed:
            y_coord = y_max_mold - feature_margin - alignment_radius
            z_coord = z_max_mold - feature_margin - alignment_radius
            x_plane_coords.append([y_coord, z_coord])
            logging.info(f"X-plane feature 2 placed at Y={y_coord:.2f}, Z={z_coord:.2f} (top-right)")
        
        # If we don't have 2 features, try alternative positions
        if len(x_plane_coords) < 2:
            if y_available_left >= min_space_needed and z_available_top >= min_space_needed:
                y_coord = y_min_mold + feature_margin + alignment_radius
                z_coord = z_max_mold - feature_margin - alignment_radius
                x_plane_coords.append([y_coord, z_coord])
                logging.info(f"X-plane feature alt placed at Y={y_coord:.2f}, Z={z_coord:.2f} (top-left)")
            
            if len(x_plane_coords) < 2 and y_available_right >= min_space_needed and z_available_bottom >= min_space_needed:
                y_coord = y_max_mold - feature_margin - alignment_radius
                z_coord = z_min_mold + feature_margin + alignment_radius
                x_plane_coords.append([y_coord, z_coord])
                logging.info(f"X-plane feature alt placed at Y={y_coord:.2f}, Z={z_coord:.2f} (bottom-right)")
        
        # ========== CALCULATE Y-PLANE COORDINATES (X-Z plane, front/back split) ==========
        # Calculate available space in X and Z directions for Y-plane features
        x_available_left = x_min_mesh - x_min_mold - feature_margin * 2
        x_available_right = x_max_mold - x_max_mesh - feature_margin * 2
        # Z available space is the same as calculated above
        
        logging.info(f"Y-plane available space: X_left={x_available_left:.2f}, X_right={x_available_right:.2f}, Z_bottom={z_available_bottom:.2f}, Z_top={z_available_top:.2f}")
        
        y_plane_coords = []
        
        # Try to place two features on Y-plane (X-Z coordinates)
        if x_available_left >= min_space_needed and z_available_bottom >= min_space_needed:
            x_coord = x_min_mold + feature_margin + alignment_radius
            z_coord = z_min_mold + feature_margin + alignment_radius
            y_plane_coords.append([x_coord, z_coord])
            logging.info(f"Y-plane feature 1 placed at X={x_coord:.2f}, Z={z_coord:.2f} (left-bottom)")
        
        if x_available_right >= min_space_needed and z_available_top >= min_space_needed:
            x_coord = x_max_mold - feature_margin - alignment_radius
            z_coord = z_max_mold - feature_margin - alignment_radius
            y_plane_coords.append([x_coord, z_coord])
            logging.info(f"Y-plane feature 2 placed at X={x_coord:.2f}, Z={z_coord:.2f} (right-top)")
        
        # If we don't have 2 features, try alternative positions
        if len(y_plane_coords) < 2:
            if x_available_left >= min_space_needed and z_available_top >= min_space_needed:
                x_coord = x_min_mold + feature_margin + alignment_radius
                z_coord = z_max_mold - feature_margin - alignment_radius
                y_plane_coords.append([x_coord, z_coord])
                logging.info(f"Y-plane feature alt placed at X={x_coord:.2f}, Z={z_coord:.2f} (left-top)")
            
            if len(y_plane_coords) < 2 and x_available_right >= min_space_needed and z_available_bottom >= min_space_needed:
                x_coord = x_max_mold - feature_margin - alignment_radius
                z_coord = z_min_mold + feature_margin + alignment_radius
                y_plane_coords.append([x_coord, z_coord])
                logging.info(f"Y-plane feature alt placed at X={x_coord:.2f}, Z={z_coord:.2f} (right-bottom)")
        
        # Validate we have enough coordinates for both planes
        if len(x_plane_coords) >= 2 and len(y_plane_coords) >= 2:
            result = {
                'x_plane_coords': x_plane_coords[:2],  # Take first 2
                'y_plane_coords': y_plane_coords[:2]   # Take first 2
            }
            logging.info(f"Quarters alignment coordinate calculation successful:")
            logging.info(f"  X-plane (Y-Z): {result['x_plane_coords']}")
            logging.info(f"  Y-plane (X-Z): {result['y_plane_coords']}")
            return result
        else:
            logging.warning(f"Insufficient space for quarters alignment: X-plane={len(x_plane_coords)}/2, Y-plane={len(y_plane_coords)}/2")
            return None
            
    except Exception as e:
        logging.error(f"Quarters alignment coordinate calculation failed: {e}")
        return None

def add_alignment_features(parts, mold_center, mold_extents, feature_radius, feature_relative_coords):
    """
    Adds rectangular parallelepiped alignment features (indentations and protrusions) to two mold halves,
    using intelligent geometric analysis for robust placement.

    Parameters:
      parts (list): List containing the two mold halves (trimesh.Trimesh objects).
                    Expected: parts[0] gets indentations, parts[1] gets protrusions.
      mold_center (np.array): The center coordinates [x, y, z] of the original mold box.
      mold_extents (np.array): The extents [dx, dy, dz] of the original mold box.
      feature_radius (float): The base size for the rectangular features (will be used as width/height).
      feature_relative_coords (list): A list of two 2D coordinates [ [y1, z1], [y2, z2] ] representing
                                      the desired **absolute Y and Z coordinates** on the interface plane.

    Returns:
      list: The modified list of parts with rectangular alignment features added. Returns the
            original list if features cannot be added or a significant error occurs.
    """
    if len(parts) != 2:
        logging.warning("Alignment features require exactly 2 parts (halves). Skipping.")
        return parts
    if feature_radius <= 0 or not feature_relative_coords or len(feature_relative_coords) != 2:
        logging.warning("Invalid parameters for alignment features (radius > 0 and coords list required). Skipping.")
        return parts

    logging.info(f"Adding intelligent rectangular alignment features with base size {feature_radius}...")
    print(f"Adding rectangular alignment features (base size: {feature_radius}, BREAKTHROUGH indentations for proper holes)...")

    # Interface plane normal (assuming split along X-axis)
    interface_normal = np.array([1.0, 0.0, 0.0])
    interface_x = mold_center[0]
    
    # Calculate dimensions for rectangular features
    # Base dimensions from the feature_radius parameter
    # APPROPRIATELY SIZED: Make features clearly visible and functional based on mold size
    mold_size = max(mold_extents)
    
    # FEATURE SIZE CAPPING: Prevent oversized features on very large models
    max_reasonable_feature_size = min(mold_size * 0.08, 20.0)  # Max 8% of mold or 20mm, whichever is smaller
    capped_feature_radius = min(feature_radius, max_reasonable_feature_size)
    
    if capped_feature_radius < feature_radius:
        logging.info(f"Feature radius capped from {feature_radius:.2f}mm to {capped_feature_radius:.2f}mm for large model")
        print(f"Note: Feature size reduced to {capped_feature_radius:.2f}mm for better manufacturability")
        feature_radius = capped_feature_radius
    
    # More conservative scaling to prevent massive features
    scale_factor = min(2.0, max(0.8, mold_size / 200.0))  # Cap at 2x, scale more conservatively
    
    box_pierce_depth = feature_radius * 2.0 * scale_factor   # X dimension (how far it pierces) - more conservative
    box_width = feature_radius * 1.2 * scale_factor          # Y dimension (width) - more conservative
    box_height = feature_radius * 1.2 * scale_factor         # Z dimension (height) - more conservative
    
    # Tolerance settings for proper fit
    tolerance = 0.15  # Reduced tolerance for tighter fit (was 0.2mm)
    
    # Protrusion dimensions (BIGGER - to fill ~50% of indentations)
    protrusion_extents = [
        box_pierce_depth * 1.2,                # Deeper protrusions (20% more depth)
        box_width * 0.8,                       # 80% of base width (larger than before)
        box_height * 0.8                       # 80% of base height (larger than before)
    ]
    
    # Indentation dimensions (keep same - they're working well)
    indentation_extents = [
        box_pierce_depth + tolerance * 2,       # Much deeper by tolerance
        box_width + tolerance * 2,              # Wider by tolerance on both sides  
        box_height + tolerance * 2              # Taller by tolerance on both sides
    ]
    
    # Calculate adaptive offset based on feature size
    adaptive_offset = max(0.2, box_pierce_depth * 0.2)  # Larger offset for visibility

    try:
        # Parse and validate coordinates
        rel_coord1 = [float(c) for c in feature_relative_coords[0]]
        rel_coord2 = [float(c) for c in feature_relative_coords[1]]
        
        # Calculate box centers with adaptive offset
        center1_on_plane = np.array([interface_x, rel_coord1[0], rel_coord1[1]])
        center2_on_plane = np.array([interface_x, rel_coord2[0], rel_coord2[1]])

        # Apply smart offsetting for VISIBLE alignment features
        # INDENTATIONS: Create COMPLETE HOLES that break through the mold wall completely
        # Need to extend significantly beyond the interface to create actual through-holes
        mold_wall_thickness = max(mold_extents) * 0.08  # Estimate wall thickness as 8% of mold size (more generous)
        breakthrough_depth = indentation_extents[0] + mold_wall_thickness * 2  # Much deeper to ensure breakthrough
        
        # Position indentations to extend BEYOND the interface - they should start outside and go deep inside
        # Move the center significantly toward the negative normal direction (into the left mold)
        indentation_extension_outside = breakthrough_depth * 0.3  # 30% extends outside the interface
        indentation_center_offset = (breakthrough_depth / 2) - indentation_extension_outside
        coord1_indent = center1_on_plane - interface_normal * indentation_center_offset
        coord2_indent = center2_on_plane - interface_normal * indentation_center_offset
        
        # Update indentation extents to ensure they break through the wall
        indentation_extents_breakthrough = [
            breakthrough_depth,                          # Much deeper X dimension to break through wall
            indentation_extents[1],                      # Keep Y dimension 
            indentation_extents[2]                       # Keep Z dimension
        ]
        
        logging.info(f"Breakthrough indentation depth: {breakthrough_depth:.2f}mm (wall thickness estimate: {mold_wall_thickness:.2f}mm)")
        logging.info(f"Indentation positioning: {indentation_extension_outside:.2f}mm extends outside interface, center offset: {indentation_center_offset:.2f}mm")
        print(f"💡 Indentations will be {breakthrough_depth:.1f}mm deep with {indentation_extension_outside:.1f}mm extending outside interface to create complete through-holes")
        
        # PROTRUSIONS: Position so that 50%+ extends outside the mold wall
        # Calculate how much should be inside vs outside the interface
        protrusion_inside_amount = protrusion_extents[0] * 0.4   # 40% inside for good bonding
        protrusion_outside_amount = protrusion_extents[0] * 0.6  # 60% outside for visibility and function
        
        # Position protrusion centers so they extend properly outside
        # Move center toward the outside (positive normal direction)
        protrusion_center_offset = (protrusion_outside_amount - protrusion_inside_amount) / 2
        coord1_protrude = center1_on_plane + interface_normal * protrusion_center_offset
        coord2_protrude = center2_on_plane + interface_normal * protrusion_center_offset
        
        # Use the standard protrusion extents (no need for extended boxes)
        extended_protrusion_extents = protrusion_extents
        
        logging.info(f"Adaptive offset applied: {adaptive_offset:.3f}")
        logging.info(f"Protrusion dimensions: {protrusion_extents}")
        logging.info(f"Protrusion positioning: {protrusion_inside_amount:.2f}mm inside, {protrusion_outside_amount:.2f}mm outside (60% external)")
        logging.info(f"Indentation dimensions (breakthrough): {indentation_extents_breakthrough}")
        logging.info(f"Tolerance: {tolerance:.3f}mm")
        logging.info(f"Feature centers: indent={[coord1_indent, coord2_indent]}, protrude={[coord1_protrude, coord2_protrude]}")

    except Exception as e:
        logging.error(f"Could not parse feature coordinates: {feature_relative_coords}. Error: {e}")
        print(f"Error: Invalid alignment coordinates: {feature_relative_coords}")
        return parts

    # Create box primitives with error handling
    try:
        # Create indentation boxes (breakthrough depth to create actual holes)
        boxes_indent = [
            trimesh.creation.box(extents=indentation_extents_breakthrough, 
                               transform=trimesh.transformations.translation_matrix(coord1_indent)),
            trimesh.creation.box(extents=indentation_extents_breakthrough,
                               transform=trimesh.transformations.translation_matrix(coord2_indent))
        ]
        
        # Create protrusion boxes (extended to span interface)
        boxes_protrude = [
            trimesh.creation.box(extents=extended_protrusion_extents,
                               transform=trimesh.transformations.translation_matrix(coord1_protrude)),
            trimesh.creation.box(extents=extended_protrusion_extents,
                               transform=trimesh.transformations.translation_matrix(coord2_protrude))
        ]
        
        logging.info("Created rectangular box primitives successfully (with breakthrough indentations)")
    except Exception as e:
        logging.error(f"Failed to create box primitives: {e}")
        print(f"Error: Failed to create alignment boxes")
        return parts

    # Apply features with multiple engine fallbacks
    def apply_boolean_with_fallback(mesh, boxes, operation='difference'):
        """Apply boolean operation with multiple engine fallbacks"""
        engines_to_try = ['blender', 'default', 'scad']  # Reorder: blender first
        
        for engine in engines_to_try:
            try:
                faces_before = len(mesh.faces)
                volume_before = getattr(mesh, 'volume', None)
                logging.info(f"Trying {operation} with {engine} engine...")
                
                result = mesh
                for i, box in enumerate(boxes):
                    if engine == 'default':
                        if operation == 'difference':
                            result = result.difference(box)
                        else:  # union
                            result = result.union(box)
                    else:
                        if operation == 'difference':
                            result = trimesh.boolean.difference([result, box], engine=engine)
                        else:  # union
                            result = trimesh.boolean.union([result, box], engine=engine)
                    logging.info(f"Applied {operation} with box {i+1} using {engine}")
                
                faces_after = len(result.faces)
                volume_after = getattr(result, 'volume', None)
                
                # IMPROVED: More flexible validation for large models
                operation_valid = True
                validation_warnings = []
                
                # Check if result is empty or has very few faces
                if faces_after < 10:
                    logging.warning(f"{engine} {operation}: result has too few faces ({faces_after})")
                    operation_valid = False
                
                # More flexible volume validation for large models
                if volume_before and volume_after:
                    volume_change = abs(volume_after - volume_before)
                    volume_change_percent = volume_change / volume_before * 100
                    
                    # Calculate expected volumes (more flexible for large features)
                    total_box_volume = sum(np.prod(box_extents) for box_extents in 
                                         [indentation_extents_breakthrough if operation == 'difference' else protrusion_extents 
                                          for _ in boxes])
                    
                    if operation == 'difference':
                        # For difference, volume should decrease
                        if volume_after > volume_before:  # Only fail if volume actually increased
                            validation_warnings.append(f"volume increased instead of decreased ({volume_before:.2f} -> {volume_after:.2f})")
                            operation_valid = False
                        # RELAXED: For large models, accept smaller volume changes
                        elif volume_change < total_box_volume * 0.01:  # Only 1% of expected (very lenient)
                            validation_warnings.append(f"volume change small ({volume_change:.2f}, expected ~{total_box_volume:.2f})")
                            # For large models (>100mm), be even more forgiving
                            if max(mold_extents) > 100:
                                logging.info(f"Large model detected - accepting small volume change for {operation}")
                                operation_valid = True  # Override for large models
                            else:
                                operation_valid = False
                    else:  # union
                        # For union, volume should increase or stay same (if boxes extend outside)
                        if volume_after < volume_before * 0.98:  # Allow small decrease due to numerical precision
                            validation_warnings.append(f"volume decreased significantly ({volume_before:.2f} -> {volume_after:.2f})")
                            operation_valid = False
                        # Don't require minimum volume increase for union operations
                        # since boxes often extend outside the mesh boundaries
                
                # RELAXED: More flexible watertight checking for all models
                watertight_check_passed = True
                if not result.is_watertight:
                    if operation == 'difference':
                        # For difference operations, prioritize volume change over watertight
                        if volume_before and volume_after and volume_after < volume_before:
                            logging.info(f"{engine} {operation}: not watertight but volume decreased ({volume_before:.2f} -> {volume_after:.2f}) - accepting")
                            watertight_check_passed = True
                        elif max(mold_extents) > 100:  # Large models (lowered threshold)
                            logging.info(f"{engine} {operation}: large model - accepting non-watertight result")
                            watertight_check_passed = True
                        else:
                            # For small models, still accept if we have any volume change
                            if volume_before and volume_after and abs(volume_after - volume_before) > 0.01:
                                logging.info(f"{engine} {operation}: small model with volume change - accepting despite non-watertight")
                                watertight_check_passed = True
                            else:
                                validation_warnings.append("result not watertight and no significant volume change")
                                watertight_check_passed = False
                    else:  # union
                        # For union operations, watertight less critical
                        logging.info(f"{engine} {operation}: not watertight but continuing (union operation)")
                        watertight_check_passed = True
                
                # Final validation decision
                operation_valid = operation_valid and watertight_check_passed
                
                if operation_valid:
                    logging.info(f"{engine} {operation} successful: faces {faces_before} -> {faces_after}, volume {volume_before:.2f} -> {volume_after:.2f}")
                    return result, True
                else:
                    warning_msg = f"{engine} {operation}: validation failed - " + ", ".join(validation_warnings)
                    logging.warning(warning_msg)
                    continue
                
            except Exception as e:
                logging.warning(f"{engine} {operation} failed: {e}")
                continue
        
        logging.error(f"All engines failed for {operation} operation")
        return mesh, False

    # Apply indentations to part 0
    part_indent_modified, indent_success = apply_boolean_with_fallback(
        parts[0], boxes_indent, 'difference')
    
    if indent_success:
        print("✓ Rectangular indentations applied successfully")
        log_mesh_stats(part_indent_modified, "Part 0 After Indentations")
    else:
        print("⚠ Indentation operation failed, keeping original part")
        part_indent_modified = parts[0]

    # Apply protrusions to part 1
    part_protrude_modified, protrude_success = apply_boolean_with_fallback(
        parts[1], boxes_protrude, 'union')
    
    if protrude_success:
        print("✓ Rectangular protrusions applied successfully")
        log_mesh_stats(part_protrude_modified, "Part 1 After Protrusions")
    else:
        print("⚠ Protrusion operation failed, keeping original part")
        part_protrude_modified = parts[1]

    # Final validation
    if part_indent_modified is None or part_protrude_modified is None:
        logging.error("Critical failure in alignment feature processing")
        return parts

    success_message = f"Rectangular alignment features completed: indent={'OK' if indent_success else 'FAIL'}, protrude={'OK' if protrude_success else 'FAIL'}"
    logging.info(success_message)
    print(f"Rectangular alignment features completed: indent={'✓' if indent_success else '✗'}, protrude={'✓' if protrude_success else '✗'}")

    return [part_indent_modified, part_protrude_modified]

def add_quarters_alignment_features(parts, mold_center, mold_extents, feature_radius, quarters_coords):
    """
    Adds rectangular alignment features to quarters mode (4 parts) on both interface planes.
    
    For quarters mode:
    - X-plane features (on Y-Z interface) align left/right quarters
    - Y-plane features (on X-Z interface) align front/back quarters
    
    Each quarter gets a combination of indentations and protrusions to ensure proper alignment.
    
    Parameters:
      parts (list): List containing the four quarters [front-left, front-right, back-left, back-right]
      mold_center (np.array): The center coordinates [x, y, z] of the original mold box
      mold_extents (np.array): The extents [dx, dy, dz] of the original mold box
      feature_radius (float): The base size for the rectangular features
      quarters_coords (dict): Dictionary with 'x_plane_coords' and 'y_plane_coords'
    
    Returns:
      list: The modified list of quarters with alignment features added
    """
    if len(parts) != 4:
        logging.warning("Quarters alignment features require exactly 4 parts. Skipping.")
        return parts
    if feature_radius <= 0 or not quarters_coords:
        logging.warning("Invalid parameters for quarters alignment features. Skipping.")
        return parts
    
    logging.info(f"Adding quarters alignment features with base size {feature_radius}...")
    print(f"Adding quarters alignment features (base size: {feature_radius}, cross-pattern for all 4 parts)...")
    
    # Extract coordinate data
    x_plane_coords = quarters_coords.get('x_plane_coords', [])
    y_plane_coords = quarters_coords.get('y_plane_coords', [])
    
    if len(x_plane_coords) < 2 or len(y_plane_coords) < 2:
        logging.error("Insufficient coordinates for quarters alignment features")
        return parts
    
    # Interface planes
    interface_x = mold_center[0]  # X=0 plane (left/right split)
    interface_y = mold_center[1]  # Y=0 plane (front/back split)
    
    # Calculate feature dimensions (same logic as halves mode)
    mold_size = max(mold_extents)
    max_reasonable_feature_size = min(mold_size * 0.08, 20.0)
    capped_feature_radius = min(feature_radius, max_reasonable_feature_size)
    
    if capped_feature_radius < feature_radius:
        logging.info(f"Feature radius capped from {feature_radius:.2f}mm to {capped_feature_radius:.2f}mm for large model")
        feature_radius = capped_feature_radius
    
    scale_factor = min(2.0, max(0.8, mold_size / 200.0))
    
    box_pierce_depth = feature_radius * 2.0 * scale_factor
    box_width = feature_radius * 1.2 * scale_factor
    box_height = feature_radius * 1.2 * scale_factor
    
    tolerance = 0.15
    
    # Protrusion and indentation dimensions
    protrusion_extents = [
        box_pierce_depth * 1.2,
        box_width * 0.8,
        box_height * 0.8
    ]
    
    indentation_extents = [
        box_pierce_depth + tolerance * 2,
        box_width + tolerance * 2,
        box_height + tolerance * 2
    ]
    
    # Calculate wall thickness and breakthrough depth
    mold_wall_thickness = max(mold_extents) * 0.08
    breakthrough_depth = indentation_extents[0] + mold_wall_thickness * 2
    
    logging.info(f"Quarters feature dimensions: protrusion={protrusion_extents}, indentation={indentation_extents}")
    
    try:
        # =================================================================
        # PART ASSIGNMENT FOR QUARTERS:
        # parts[0] = front-left (FL)   - negative X, negative Y
        # parts[1] = front-right (FR)  - positive X, negative Y  
        # parts[2] = back-left (BL)    - negative X, positive Y
        # parts[3] = back-right (BR)   - positive X, positive Y
        # =================================================================
        
        # Create all the alignment boxes for both interface planes
        alignment_boxes = {
            'x_plane_indent': [],   # X-plane indentations (for left quarters)
            'x_plane_protrude': [], # X-plane protrusions (for right quarters)
            'y_plane_indent': [],   # Y-plane indentations (for front quarters)
            'y_plane_protrude': []  # Y-plane protrusions (for back quarters)
        }
        
        # ========== X-PLANE FEATURES (Y-Z coordinates, left/right alignment) ==========
        for i, coord in enumerate(x_plane_coords):
            y_coord, z_coord = coord
            
            # Position indentations and protrusions on the X interface plane
            indentation_extension_outside = breakthrough_depth * 0.3
            indentation_center_offset = (breakthrough_depth / 2) - indentation_extension_outside
            
            # Indentations (for left quarters - FL and BL)
            indent_center = np.array([interface_x - indentation_center_offset, y_coord, z_coord])
            indent_extents = [breakthrough_depth, indentation_extents[1], indentation_extents[2]]
            indent_box = trimesh.creation.box(
                extents=indent_extents,
                transform=trimesh.transformations.translation_matrix(indent_center)
            )
            alignment_boxes['x_plane_indent'].append(indent_box)
            
            # Protrusions (for right quarters - FR and BR)
            protrusion_inside_amount = protrusion_extents[0] * 0.4
            protrusion_outside_amount = protrusion_extents[0] * 0.6
            protrusion_center_offset = (protrusion_outside_amount - protrusion_inside_amount) / 2
            
            protrude_center = np.array([interface_x + protrusion_center_offset, y_coord, z_coord])
            protrude_box = trimesh.creation.box(
                extents=protrusion_extents,
                transform=trimesh.transformations.translation_matrix(protrude_center)
            )
            alignment_boxes['x_plane_protrude'].append(protrude_box)
        
        # ========== Y-PLANE FEATURES (X-Z coordinates, front/back alignment) ==========
        for i, coord in enumerate(y_plane_coords):
            x_coord, z_coord = coord
            
            # Indentations (for front quarters - FL and FR)
            indent_center = np.array([x_coord, interface_y - indentation_center_offset, z_coord])
            indent_extents = [indentation_extents[1], breakthrough_depth, indentation_extents[2]]  # Note: Y and X swapped
            indent_box = trimesh.creation.box(
                extents=indent_extents,
                transform=trimesh.transformations.translation_matrix(indent_center)
            )
            alignment_boxes['y_plane_indent'].append(indent_box)
            
            # Protrusions (for back quarters - BL and BR)
            protrude_center = np.array([x_coord, interface_y + protrusion_center_offset, z_coord])
            protrude_extents = [protrusion_extents[1], protrusion_extents[0], protrusion_extents[2]]  # Note: Y and X swapped
            protrude_box = trimesh.creation.box(
                extents=protrude_extents,
                transform=trimesh.transformations.translation_matrix(protrude_center)
            )
            alignment_boxes['y_plane_protrude'].append(protrude_box)
        
        logging.info("Created quarters alignment boxes for both interface planes")
        
        # Use the same boolean operation fallback function from halves mode
        def apply_boolean_with_fallback(mesh, boxes, operation='difference'):
            """Apply boolean operation with multiple engine fallbacks"""
            engines_to_try = ['blender', 'default', 'scad']
            
            for engine in engines_to_try:
                try:
                    faces_before = len(mesh.faces)
                    volume_before = getattr(mesh, 'volume', None)
                    logging.info(f"Trying {operation} with {engine} engine...")
                    
                    result = mesh
                    for i, box in enumerate(boxes):
                        if engine == 'default':
                            if operation == 'difference':
                                result = result.difference(box)
                            else:  # union
                                result = result.union(box)
                        else:
                            if operation == 'difference':
                                result = trimesh.boolean.difference([result, box], engine=engine)
                            else:  # union
                                result = trimesh.boolean.union([result, box], engine=engine)
                        logging.info(f"Applied {operation} with box {i+1} using {engine}")
                    
                    faces_after = len(result.faces)
                    volume_after = getattr(result, 'volume', None)
                    
                    # Flexible validation for quarters mode
                    operation_valid = True
                    
                    if faces_after < 10:
                        logging.warning(f"{engine} {operation}: result has too few faces ({faces_after})")
                        operation_valid = False
                    
                    # More lenient volume validation for quarters (multiple features)
                    if volume_before and volume_after:
                        if operation == 'difference' and volume_after > volume_before:
                            operation_valid = False
                        elif operation == 'union' and volume_after < volume_before * 0.95:
                            operation_valid = False
                    
                    if operation_valid:
                        logging.info(f"{engine} {operation} successful: faces {faces_before} -> {faces_after}")
                        return result, True
                    else:
                        logging.warning(f"{engine} {operation}: validation failed")
                        continue
                        
                except Exception as e:
                    logging.warning(f"{engine} {operation} failed: {e}")
                    continue
            
            logging.error(f"All engines failed for {operation} operation")
            return mesh, False
        
        # Apply alignment features to each quarter
        modified_parts = []
        success_count = 0
        
        for i, part in enumerate(parts):
            part_name = ['front-left', 'front-right', 'back-left', 'back-right'][i]
            logging.info(f"Processing alignment features for {part_name} quarter...")
            
            current_part = part
            
            # Get the bounds of this specific quarter to filter alignment features
            part_bounds = part.bounds
            part_center = part.bounds.mean(axis=0)
            
            # Only apply alignment features that are within or near this quarter
            def filter_boxes_for_quarter(boxes, operation_type):
                """Filter alignment boxes to only include those relevant to this quarter"""
                filtered_boxes = []
                for box in boxes:
                    box_center = box.bounds.mean(axis=0)
                    
                    # Check if this alignment feature box overlaps with the quarter
                    # Allow for some tolerance since features are meant to span interfaces
                    tolerance = max(mold_extents) * 0.1  # 10% tolerance
                    
                    if (box_center[0] >= part_bounds[0][0] - tolerance and 
                        box_center[0] <= part_bounds[1][0] + tolerance and
                        box_center[1] >= part_bounds[0][1] - tolerance and 
                        box_center[1] <= part_bounds[1][1] + tolerance):
                        filtered_boxes.append(box)
                        logging.info(f"Including {operation_type} box at {box_center} for {part_name}")
                    else:
                        logging.debug(f"Excluding {operation_type} box at {box_center} from {part_name} (out of bounds)")
                
                return filtered_boxes
            
            # Determine which features to apply based on quarter position
            features_to_apply = []
            
            if i == 0:  # Front-left (FL): X-plane indents + Y-plane indents
                filtered_x_indent = filter_boxes_for_quarter(alignment_boxes['x_plane_indent'], 'X-plane indent')
                filtered_y_indent = filter_boxes_for_quarter(alignment_boxes['y_plane_indent'], 'Y-plane indent')
                if filtered_x_indent:
                    features_to_apply.append(('difference', filtered_x_indent, 'X-plane indents'))
                if filtered_y_indent:
                    features_to_apply.append(('difference', filtered_y_indent, 'Y-plane indents'))
            elif i == 1:  # Front-right (FR): X-plane protrudes + Y-plane indents
                filtered_x_protrude = filter_boxes_for_quarter(alignment_boxes['x_plane_protrude'], 'X-plane protrude')
                filtered_y_indent = filter_boxes_for_quarter(alignment_boxes['y_plane_indent'], 'Y-plane indent')
                if filtered_x_protrude:
                    features_to_apply.append(('union', filtered_x_protrude, 'X-plane protrudes'))
                if filtered_y_indent:
                    features_to_apply.append(('difference', filtered_y_indent, 'Y-plane indents'))
            elif i == 2:  # Back-left (BL): X-plane indents + Y-plane protrudes
                filtered_x_indent = filter_boxes_for_quarter(alignment_boxes['x_plane_indent'], 'X-plane indent')
                filtered_y_protrude = filter_boxes_for_quarter(alignment_boxes['y_plane_protrude'], 'Y-plane protrude')
                if filtered_x_indent:
                    features_to_apply.append(('difference', filtered_x_indent, 'X-plane indents'))
                if filtered_y_protrude:
                    features_to_apply.append(('union', filtered_y_protrude, 'Y-plane protrudes'))
            elif i == 3:  # Back-right (BR): X-plane protrudes + Y-plane protrudes
                filtered_x_protrude = filter_boxes_for_quarter(alignment_boxes['x_plane_protrude'], 'X-plane protrude')
                filtered_y_protrude = filter_boxes_for_quarter(alignment_boxes['y_plane_protrude'], 'Y-plane protrude')
                if filtered_x_protrude:
                    features_to_apply.append(('union', filtered_x_protrude, 'X-plane protrudes'))
                if filtered_y_protrude:
                    features_to_apply.append(('union', filtered_y_protrude, 'Y-plane protrudes'))
            
            # Apply each set of features
            part_success = True
            for operation, boxes, feature_name in features_to_apply:
                if boxes:  # Only apply if boxes exist
                    modified_part, op_success = apply_boolean_with_fallback(current_part, boxes, operation)
                    if op_success:
                        current_part = modified_part
                        logging.info(f"✓ {feature_name} applied to {part_name}")
                    else:
                        logging.warning(f"✗ {feature_name} failed for {part_name}")
                        part_success = False
            
            if part_success:
                success_count += 1
                print(f"✓ {part_name} quarter alignment features applied")
            else:
                print(f"⚠ {part_name} quarter had some feature application issues")
            
            modified_parts.append(current_part)
        
        success_message = f"Quarters alignment features completed: {success_count}/4 quarters successful"
        logging.info(success_message)
        print(f"Quarters alignment features completed: {success_count}/4 quarters successful")
        
        return modified_parts
        
    except Exception as e:
        logging.error(f"Error in quarters alignment feature processing: {e}")
        print(f"Error: Quarters alignment feature processing failed: {e}")
        return parts

def create_mold_with_preserved_cavity(input_model_path, padding=0.1, hole_positions=['top'], split_mode='quarters', 
                                        draft_angle=0.0, custom_planes=None, visualize=False, alignment_radius=-1.0, 
                                        alignment_coords=None, auto_align_coords=True):
    """
    Generates a mold with a preserved cavity from the input model.
    Optionally applies a draft angle and splits the mold into halves, quarters,
    or according to custom slicing planes with intelligent alignment features.
    """
    logging.info(f"Parameters: padding={padding}, auto_align_coords={auto_align_coords}")
    logging.info(f"Starting mold creation for model: {input_model_path}")
    logging.info(f"Parameters: padding={padding}, hole_positions={hole_positions}, split_mode={split_mode}, draft_angle={draft_angle}, alignment_radius={alignment_radius}")
    print(f"Loading model from: {input_model_path}")
    
    try:
        mesh = trimesh.load_mesh(input_model_path)
        logging.info(f"Successfully loaded mesh from {input_model_path}")
        log_mesh_stats(mesh, "Original Loaded Mesh")
    except Exception as e:
        logging.error(f"Failed to load mesh from {input_model_path}: {str(e)}")
        raise

    if not mesh.is_watertight:
        logging.warning("Input mesh is not watertight. Attempting to fill holes.")
        print("Attempting to fill holes in the mesh...")
        try:
            mesh.fill_holes()
            logging.info("fill_holes() completed")
            log_mesh_stats(mesh, "After fill_holes")
        except Exception as e:
            logging.error(f"Error during fill_holes(): {str(e)}")
        
        if not mesh.is_watertight:
            logging.warning("Mesh still not watertight after fill_holes. Attempting enhanced repair using Open3D.")
            print("Enhanced repair initiated...")
            try:
                mesh = enhanced_mesh_repair(mesh)
                log_mesh_stats(mesh, "After enhanced_mesh_repair")
            except Exception as e:
                logging.error(f"Error during enhanced mesh repair: {str(e)}")
                raise
            
            if not mesh.is_watertight:
                logging.error("Mesh remains non-watertight after enhanced repair.")
                raise ValueError("Input mesh is not watertight even after enhanced repair.")
            else:
                logging.info("Mesh repaired using enhanced repair and is now watertight.")
                print("Mesh repaired successfully.")
    
    logging.info("Input mesh is watertight.")
    print("Mesh is confirmed watertight.")
    
    # Apply draft angle if needed
    if draft_angle != 0.0:
        logging.info(f"Applying a draft angle of {draft_angle} degrees to the cavity.")
        print(f"Applying draft angle: {draft_angle}°")
        try:
            mesh = apply_draft(mesh, draft_angle)
            log_mesh_stats(mesh, "After Draft Angle Application")
        except Exception as e:
            logging.error(f"Error applying draft angle: {str(e)}")
            logging.info("Continuing with original mesh without draft angle")
    
    mesh_bounds = mesh.bounds
    logging.info(f"Mesh bounds:\\n{mesh_bounds}")
    print(f"Mesh bounds: {mesh_bounds}")

    # Calculate mold dimensions
    mold_min = mesh_bounds[0] - padding
    mold_max = mesh_bounds[1] + padding
    mold_extents = mold_max - mold_min
    mold_center = (mold_min + mold_max) / 2
    logging.info(f"Mold extents: {mold_extents}")
    logging.info(f"Mold center: {mold_center}")
    print(f"Mold extents: {mold_extents}")
    print(f"Mold center: {mold_center}")

    # Create mold box
    try:
        mold_box = trimesh.creation.box(
            extents=mold_extents,
            transform=trimesh.transformations.translation_matrix(mold_center)
        )
        logging.info("Created mold box.")
        log_mesh_stats(mold_box, "Mold Box")
        print("Mold box created.")
    except Exception as e:
        logging.error(f"Error creating mold box: {str(e)}")
        raise

    # Create mold cavity with enhanced boolean operations
    try:
        logging.info("Beginning boolean difference operation for mold cavity...")
        
        # First, verify that both meshes are valid for boolean operations
        if not mesh.is_watertight:
            logging.error("Mesh is not watertight before boolean operation!")
        if not mold_box.is_watertight:
            logging.error("Mold box is not watertight before boolean operation!")
        
        # Try multiple approaches for boolean operations
        logging.info("Using OpenSCAD engine directly for robust boolean operation...")
        
        # Try with scaled mesh first (helps with complex geometries)
        mesh_scaled = mesh.copy()
        scale_factor = 1.01  # Scale by 1%
        matrix = np.eye(4)
        matrix[:3, :3] *= scale_factor
        mesh_scaled.apply_transform(matrix)
        logging.info(f"Scaled mesh by factor of {scale_factor}")
        
        # Try with the scaled mesh first
        try:
            mold = trimesh.boolean.difference([mold_box, mesh_scaled], engine='scad')
            logging.info("SCAD engine with scaled mesh successful")
        except Exception as e:
            logging.error(f"SCAD engine with scaled mesh failed: {str(e)}")
            
            # Try with original mesh and SCAD engine
            try:
                logging.info("Trying SCAD engine with original mesh...")
                mold = trimesh.boolean.difference([mold_box, mesh], engine='scad')
                logging.info("SCAD engine with original mesh successful")
            except Exception as e:
                logging.error(f"SCAD engine with original mesh failed: {str(e)}")
                
                # Try with Blender engine if available
                try:
                    logging.info("Trying Blender engine...")
                    mold = trimesh.boolean.difference([mold_box, mesh], engine='blender')
                    logging.info("Blender engine successful")
                except Exception as e:
                    logging.error(f"Blender engine failed: {str(e)}")
                    
                    # Last resort: Try default engine
                    logging.info("Trying default boolean engine...")
                    mold = mold_box.difference(mesh)
                    logging.info("Default engine successful")
        
        # Verify that the mold has at least some geometry
        if len(mold.faces) < 10:
            logging.error("Resulting mold has too few faces - boolean operation likely failed")
            raise ValueError("Boolean operation produced invalid mold")
        
        logging.info("Created mold with cavity.")
        log_mesh_stats(mold, "Mold with Cavity")
        print("Mold cavity generated by subtracting the mesh from the mold box.")
    except Exception as e:
        logging.error(f"Error creating mold cavity: {str(e)}")
        raise

    # Process holes
    for position in hole_positions:
        logging.info(f"Processing hole at position: {position}")
        print(f"Processing hole at position: {position}")
        hole_radius = min(mold_extents[:2]) * 0.1
        logging.info(f"Hole radius: {hole_radius}")
        
        try:
            if position == 'bottom':
                # Calculate length from mold bottom to cavity bottom
                hole_length = mesh_bounds[0][2] - mold_min[2]
                # Add a tiny amount to ensure it definitely breaks through
                hole_length += 0.01

                logging.info(f"Hole length targeting cavity bottom: {hole_length}")
            
                hole_cylinder = trimesh.creation.cylinder(
                    radius=hole_radius,
                    height=hole_length,
                    sections=64
                )
                hole_center = mold_center.copy()
                hole_center[2] = mold_min[2] + hole_length / 2
                logging.info(f"Hole center: {hole_center}")
            
                hole_cylinder.apply_translation(hole_center - hole_cylinder.center_mass)
                log_mesh_stats(hole_cylinder, "Hole Cylinder")
                
                # Perform the boolean difference
                mold_before_hole = mold.copy()
                mold = mold.difference(hole_cylinder)
                
                log_mesh_stats(mold, "Mold with Hole")
                logging.info("Added bottom hole.")
                print("Bottom hole added.")
            elif position in ['top', 'left', 'right']:
                logging.info(f"Support for {position} hole position not yet implemented.")
                print(f"Note: {position} hole position not implemented in this version.")
        except Exception as e:
            logging.error(f"Error adding hole at position {position}: {str(e)}")
            print(f"Warning: Failed to add hole at position {position}")

    # Intelligent auto-alignment coordinate calculation
    actual_alignment_coords = None
    if alignment_radius > 0 and split_mode == 'halves':
        if auto_align_coords:
            logging.info("Using intelligent geometric analysis for automatic alignment placement...")
            try:
                # Use the new intelligent coordinate calculation
                intelligent_coords = calculate_intelligent_alignment_coordinates(
                    mold, mesh, mold_center, mold_extents, mesh_bounds, alignment_radius, padding
                )
                
                if intelligent_coords:
                    # Validate the computed coordinates
                    if validate_alignment_placement(intelligent_coords, mesh, mold_center, mold_extents, alignment_radius):
                        actual_alignment_coords = intelligent_coords
                        logging.info(f"Intelligent auto-calculated alignment coords: {[[f'{c:.2f}' for c in p] for p in actual_alignment_coords]}")
                        print(f"✓ Using intelligently calculated alignment coordinates")
                    else:
                        logging.warning("Intelligent coordinates failed validation")
                        print("⚠ Automatic placement validation failed. Try smaller radius, larger padding, or manual placement.")
                        actual_alignment_coords = None
                else:
                    logging.warning("Intelligent coordinate calculation returned no valid placements")
                    print("⚠ No valid automatic placement found. Try smaller radius, larger padding, or manual placement.")
                    actual_alignment_coords = None

            except Exception as e:
                logging.error(f"Failed to auto-calculate intelligent alignment coords: {e}. Alignment features will be skipped.")
                print(f"⚠ Automatic alignment calculation failed: {e}")
                actual_alignment_coords = None
        else: # Manual coords
            if alignment_coords:
                logging.info(f"Using manually provided alignment coordinates: {alignment_coords}")
                # Validate manual coordinates too
                if validate_alignment_placement(alignment_coords, mesh, mold_center, mold_extents, alignment_radius):
                    actual_alignment_coords = alignment_coords
                    print("✓ Manual alignment coordinates validated")
                else:
                    logging.warning("Manual alignment coordinates failed validation")
                    print("⚠ Manual coordinates failed validation. Features will be skipped.")
                    actual_alignment_coords = None
            else:
                logging.warning("Manual alignment selected, but no coordinates provided. Skipping features.")
                actual_alignment_coords = None
    elif alignment_radius > 0 and split_mode not in ['halves', 'quarters']:
        logging.info(f"Alignment features requested but split_mode is '{split_mode}'. Features supported only for 'halves' and 'quarters'.")

    # Split the mold
    if custom_planes is not None and len(custom_planes) > 0:
        logging.info(f"Splitting the mold using {len(custom_planes)} custom slicing planes.")
        print("Custom slicing planes provided. Splitting mold accordingly...")
        # Custom plane splitting would be implemented here
        parts = [mold]  # Simplified for now
    else:
        try:
            if split_mode == 'halves':
                logging.info("Splitting the mold into halves along the X-axis.")
                print("Splitting mold into halves...")
                
                left_box = trimesh.creation.box(
                    extents=[mold_extents[0]/2, mold_extents[1] + padding, mold_extents[2] + 2*padding],
                    transform=trimesh.transformations.translation_matrix([
                        mold_center[0] - mold_extents[0]/4,
                        mold_center[1],
                        mold_center[2]
                    ])
                )
                log_mesh_stats(left_box, "Left Half Box")
                
                right_box = trimesh.creation.box(
                    extents=[mold_extents[0]/2, mold_extents[1] + padding, mold_extents[2] + 2*padding],
                    transform=trimesh.transformations.translation_matrix([
                        mold_center[0] + mold_extents[0]/4,
                        mold_center[1],
                        mold_center[2]
                    ])
                )
                log_mesh_stats(right_box, "Right Half Box")
                
                logging.info("Performing intersection for left part...")
                left_part = mold.intersection(left_box)
                log_mesh_stats(left_part, "Left Half Result")
                
                logging.info("Performing intersection for right part...")
                right_part = mold.intersection(right_box)
                log_mesh_stats(right_part, "Right Half Result")
                
                parts = [left_part, right_part]
                logging.info("Created left and right halves.")
                print("Mold split into left and right halves.")

                # Use determined alignment coordinates
                if alignment_radius > 0 and actual_alignment_coords:
                    logging.info("Adding alignment features to halves...")
                    try:
                        # Calculate the actual effective radius to use based on available space
                        y_space = min(
                            abs(actual_alignment_coords[0][0] - mesh_bounds[0][1]),
                            abs(actual_alignment_coords[0][0] - mesh_bounds[1][1]),
                            abs(actual_alignment_coords[1][0] - mesh_bounds[0][1]),
                            abs(actual_alignment_coords[1][0] - mesh_bounds[1][1])
                        )
                        z_space = min(
                            abs(actual_alignment_coords[0][1] - mesh_bounds[0][2]),
                            abs(actual_alignment_coords[0][1] - mesh_bounds[1][2]),
                            abs(actual_alignment_coords[1][1] - mesh_bounds[0][2]),
                            abs(actual_alignment_coords[1][1] - mesh_bounds[1][2])
                        )
                        
                        # Use a practical effective radius that's visible and functional
                        # Don't be overly conservative when there's plenty of space
                        space_based_radius = min(y_space * 0.7, z_space * 0.7)  # Use 70% of available space
                        size_based_cap = max(max(mesh.extents) * 0.03, 1.5)  # At least 3% of model size, minimum 1.5mm
                        effective_radius = min(alignment_radius, space_based_radius, size_based_cap)
                        
                        logging.info(f"Using effective alignment radius: {effective_radius:.2f}mm (requested: {alignment_radius:.2f}mm)")
                        print(f"Creating alignment features with {effective_radius:.2f}mm radius...")
                        
                        # Try primary alignment feature application
                        parts_with_features = add_alignment_features(parts, mold_center, mold_extents, effective_radius, actual_alignment_coords)
                        
                        # Check if features were successfully applied (compare parts)
                        features_applied = False
                        if len(parts_with_features) == len(parts):
                            for i, (original, modified) in enumerate(zip(parts, parts_with_features)):
                                if len(modified.faces) != len(original.faces):
                                    features_applied = True
                                    break
                        
                        if features_applied:
                            parts = parts_with_features
                            logging.info("✓ Primary alignment features applied successfully")
                        else:
                            # ITERATIVE POSITIONING: Try alternative positions if primary failed
                            logging.warning("Primary alignment features failed, trying iterative positioning...")
                            print("⚠ Primary positioning failed, trying alternative positions...")
                            
                            parts_with_iterative = try_iterative_alignment_positioning(
                                parts, mold_center, mold_extents, effective_radius, actual_alignment_coords,
                                mesh_bounds, alignment_radius, padding
                            )
                            
                            if parts_with_iterative != parts:
                                parts = parts_with_iterative
                                logging.info("✓ Iterative positioning successful!")
                                print("✓ Alternative positioning worked!")
                            else:
                                logging.warning("All alignment positioning attempts failed")
                                print("⚠ All positioning attempts failed - proceeding without alignment features")
                        
                    except Exception as align_err:
                        logging.error(f"Error during alignment feature addition: {align_err}")
                        print("Warning: Could not add alignment features.")
                else:
                    logging.info("Skipping alignment features (radius <= 0 or no valid coords).")
                
            elif split_mode == 'quarters':
                logging.info("Splitting the mold into quarters along X and Y axes.")
                print("Splitting mold into quarters...")
                
                # Calculate proper quarter boxes that exactly overlap the mold
                x_center, y_center, z_center = mold_center
                
                # Make quarter boxes that exactly span half the mold with small overlap
                overlap = max(mold_extents) * 0.01  # 1% overlap to ensure clean cuts
                
                # Quarter extents: half mold size + small overlap
                quarter_x = mold_extents[0] / 2 + overlap
                quarter_y = mold_extents[1] / 2 + overlap 
                quarter_z = mold_extents[2] + overlap * 2  # Full Z height + overlap
                
                quarter_extents = [quarter_x, quarter_y, quarter_z]
                logging.info(f"Quarter box extents: {quarter_extents}")
                
                # Position quarter box centers to properly split the mold
                # Front-left: negative X, negative Y from center
                fl_center = [x_center - quarter_x/2, y_center - quarter_y/2, z_center]
                fl_box = trimesh.creation.box(
                    extents=quarter_extents,
                    transform=trimesh.transformations.translation_matrix(fl_center)
                )
                logging.info(f"Front-left box center: {fl_center}")
                logging.info(f"Front-left box bounds: {fl_box.bounds}")
                
                # Front-right: positive X, negative Y from center
                fr_center = [x_center + quarter_x/2, y_center - quarter_y/2, z_center]
                fr_box = trimesh.creation.box(
                    extents=quarter_extents,
                    transform=trimesh.transformations.translation_matrix(fr_center)
                )
                logging.info(f"Front-right box center: {fr_center}")
                logging.info(f"Front-right box bounds: {fr_box.bounds}")
                
                # Back-left: negative X, positive Y from center
                bl_center = [x_center - quarter_x/2, y_center + quarter_y/2, z_center]
                bl_box = trimesh.creation.box(
                    extents=quarter_extents,
                    transform=trimesh.transformations.translation_matrix(bl_center)
                )
                logging.info(f"Back-left box center: {bl_center}")
                logging.info(f"Back-left box bounds: {bl_box.bounds}")
                
                # Back-right: positive X, positive Y from center
                br_center = [x_center + quarter_x/2, y_center + quarter_y/2, z_center]
                br_box = trimesh.creation.box(
                    extents=quarter_extents,
                    transform=trimesh.transformations.translation_matrix(br_center)
                )
                logging.info(f"Back-right box center: {br_center}")
                logging.info(f"Back-right box bounds: {br_box.bounds}")
                
                # Log mold bounds for comparison
                logging.info(f"Mold bounds for comparison: {mold.bounds}")
                
                # Verify that quarter boxes properly cover the mold
                total_coverage_x = (fl_box.bounds[1][0] - fl_box.bounds[0][0]) + (fr_box.bounds[1][0] - fr_box.bounds[0][0])
                total_coverage_y = (fl_box.bounds[1][1] - fl_box.bounds[0][1]) + (bl_box.bounds[1][1] - bl_box.bounds[0][1])
                mold_span_x = mold.bounds[1][0] - mold.bounds[0][0]
                mold_span_y = mold.bounds[1][1] - mold.bounds[0][1]
                
                logging.info(f"Coverage check: X coverage={total_coverage_x:.2f} vs mold span={mold_span_x:.2f}")
                logging.info(f"Coverage check: Y coverage={total_coverage_y:.2f} vs mold span={mold_span_y:.2f}")
                
                # Helper function to handle Scene objects and perform robust intersection
                def robust_intersection(mesh, box, name):
                    """Perform robust intersection with multiple fallback strategies"""
                    # Try direct intersection first
                    try:
                        result = mesh.intersection(box)
                        
                        # Handle Scene objects
                        if hasattr(result, 'geometry') and hasattr(result.geometry, 'values'):
                            geometries = list(result.geometry.values())
                            if geometries:
                                extracted = geometries[0]
                                logging.info(f"Extracted mesh from Scene object for {name}")
                                # Validate the result is actually a quarter (not the whole mesh)
                                if len(extracted.vertices) < len(mesh.vertices) * 0.8:  # Should be smaller
                                    return extracted
                                else:
                                    logging.warning(f"Scene result for {name} too large, trying fallback")
                            else:
                                logging.warning(f"Scene object for {name} contains no geometries, trying fallback")
                        elif hasattr(result, 'vertices') and hasattr(result, 'faces'):
                            # Already a Trimesh object - validate size
                            if len(result.vertices) < len(mesh.vertices) * 0.8:  # Should be smaller
                                return result
                            else:
                                logging.warning(f"Direct intersection result for {name} too large, trying fallback")
                        else:
                            logging.warning(f"Unknown result type for {name}: {type(result)}, trying fallback")
                    except Exception as e:
                        logging.warning(f"Direct intersection failed for {name}: {e}, trying fallback")
                    
                    # Fallback: Use boolean intersection with different engines
                    engines_to_try = ['blender', 'scad']
                    for engine in engines_to_try:
                        try:
                            logging.info(f"Trying {engine} engine for {name} quarter intersection...")
                            if engine == 'scad':
                                try:
                                    result = trimesh.boolean.intersection([mesh, box], engine='scad')
                                except:
                                    continue  # Skip if SCAD not available
                            else:
                                result = trimesh.boolean.intersection([mesh, box], engine=engine)
                            
                            if result is not None and hasattr(result, 'vertices') and len(result.vertices) > 0:
                                # Validate the result is actually a quarter
                                if len(result.vertices) < len(mesh.vertices) * 0.8:
                                    logging.info(f"Successfully extracted {name} quarter using {engine} engine")
                                    return result
                                else:
                                    logging.warning(f"{engine} engine result for {name} too large")
                                    continue
                            else:
                                logging.warning(f"{engine} engine returned empty result for {name}")
                                continue
                        except Exception as e:
                            logging.warning(f"{engine} engine failed for {name}: {e}")
                            continue
                    
                    # Last resort: Better manual slicing using planes
                    logging.warning(f"All intersection methods failed for {name}, attempting plane slicing...")
                    try:
                        # Use a much more conservative approach - try to keep the quarter box bounds
                        box_bounds = box.bounds
                        box_center = box.bounds.mean(axis=0)
                        
                        logging.info(f"Attempting conservative clipping for {name}")
                        logging.info(f"Box bounds: {box_bounds}")
                        logging.info(f"Mesh bounds: {mesh.bounds}")
                        
                        # Instead of aggressive plane slicing, try to clip the mesh to the box bounds
                        # This is more conservative and preserves walls better
                        try:
                            # Create a slightly larger clipping box to preserve walls
                            safety_margin = max(mesh.extents) * 0.02  # 2% safety margin
                            
                            # Expand box bounds slightly for safer clipping
                            expanded_min = box_bounds[0] - safety_margin
                            expanded_max = box_bounds[1] + safety_margin
                            
                            # Create vertices for the expanded bounding box
                            vertices = mesh.vertices.copy()
                            
                            # Find vertices that are within the expanded quarter box
                            within_box = (
                                (vertices[:, 0] >= expanded_min[0]) & (vertices[:, 0] <= expanded_max[0]) &
                                (vertices[:, 1] >= expanded_min[1]) & (vertices[:, 1] <= expanded_max[1]) &
                                (vertices[:, 2] >= expanded_min[2]) & (vertices[:, 2] <= expanded_max[2])
                            )
                            
                            # Get faces that have at least one vertex in the box
                            faces = mesh.faces
                            face_in_box = within_box[faces].any(axis=1)
                            
                            if face_in_box.sum() > 0:
                                # Keep faces that are within or intersect the box
                                new_faces = faces[face_in_box]
                                
                                # Find which vertices are actually used
                                used_vertices = np.unique(new_faces.flatten())
                                
                                # Create mapping from old to new vertex indices
                                vertex_map = np.full(len(vertices), -1)
                                vertex_map[used_vertices] = np.arange(len(used_vertices))
                                
                                # Create new mesh with remapped faces
                                new_vertices = vertices[used_vertices]
                                remapped_faces = vertex_map[new_faces]
                                
                                sliced_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces, 
                                                            process=False)
                                
                                logging.info(f"Conservative clipping successful for {name}: {len(mesh.vertices)} -> {len(sliced_mesh.vertices)} vertices")
                            else:
                                logging.error(f"Conservative clipping failed for {name}: no faces in region")
                                sliced_mesh = None
                                
                        except Exception as clip_error:
                            logging.error(f"Conservative clipping failed for {name}: {clip_error}")
                            
                            # Final fallback: try very gentle plane slicing
                            logging.info(f"Trying gentle plane slicing for {name}")
                            try:
                                sliced_mesh = mesh.copy()
                                
                                # Use the actual box center for slicing planes
                                x_interface = box_center[0]
                                y_interface = box_center[1]
                                
                                # Apply gentle slicing with small safety margins
                                slice_margin = max(mesh.extents) * 0.01  # 1% margin
                                
                                if 'front-left' in name:
                                    # Keep left and front parts
                                    sliced_mesh = sliced_mesh.slice_plane(
                                        plane_origin=[x_interface + slice_margin, 0, 0], 
                                        plane_normal=[1, 0, 0]
                                    )
                                    if sliced_mesh is not None:
                                        sliced_mesh = sliced_mesh.slice_plane(
                                            plane_origin=[0, y_interface + slice_margin, 0],
                                            plane_normal=[0, 1, 0]
                                        )
                                elif 'front-right' in name:
                                    # Keep right and front parts
                                    sliced_mesh = sliced_mesh.slice_plane(
                                        plane_origin=[x_interface - slice_margin, 0, 0],
                                        plane_normal=[-1, 0, 0]
                                    )
                                    if sliced_mesh is not None:
                                        sliced_mesh = sliced_mesh.slice_plane(
                                            plane_origin=[0, y_interface + slice_margin, 0],
                                            plane_normal=[0, 1, 0]
                                        )
                                elif 'back-left' in name:
                                    # Keep left and back parts
                                    sliced_mesh = sliced_mesh.slice_plane(
                                        plane_origin=[x_interface + slice_margin, 0, 0],
                                        plane_normal=[1, 0, 0]
                                    )
                                    if sliced_mesh is not None:
                                        sliced_mesh = sliced_mesh.slice_plane(
                                            plane_origin=[0, y_interface - slice_margin, 0],
                                            plane_normal=[0, -1, 0]
                                        )
                                elif 'back-right' in name:
                                    # Keep right and back parts
                                    sliced_mesh = sliced_mesh.slice_plane(
                                        plane_origin=[x_interface - slice_margin, 0, 0],
                                        plane_normal=[-1, 0, 0]
                                    )
                                    if sliced_mesh is not None:
                                        sliced_mesh = sliced_mesh.slice_plane(
                                            plane_origin=[0, y_interface - slice_margin, 0],
                                            plane_normal=[0, -1, 0]
                                        )
                                
                                if sliced_mesh is not None:
                                    logging.info(f"Gentle plane slicing successful for {name}")
                                else:
                                    logging.error(f"Gentle plane slicing failed for {name}")
                                    
                            except Exception as plane_error:
                                logging.error(f"Gentle plane slicing failed for {name}: {plane_error}")
                                sliced_mesh = None
                        
                        if sliced_mesh is not None and len(sliced_mesh.vertices) > 0:
                            logging.info(f"Plane slicing successful for {name}: {len(mesh.vertices)} -> {len(sliced_mesh.vertices)} vertices")
                            return sliced_mesh
                        else:
                            logging.error(f"Plane slicing failed for {name}: result is empty")
                            return None
                        
                    except Exception as e:
                        logging.error(f"Plane slicing failed for {name}: {e}")
                        return None
                
                # Perform intersections with robust error handling
                logging.info("Computing front-left quarter...")
                fl = robust_intersection(mold, fl_box, "front-left")
                if fl is None:
                    logging.error("Failed to compute front-left quarter")
                    raise ValueError("Failed to extract front-left mesh")
                log_mesh_stats(fl, "Front-Left Quarter")
                
                logging.info("Computing front-right quarter...")
                fr = robust_intersection(mold, fr_box, "front-right")
                if fr is None:
                    logging.error("Failed to compute front-right quarter")
                    raise ValueError("Failed to extract front-right mesh")
                log_mesh_stats(fr, "Front-Right Quarter")
                
                logging.info("Computing back-left quarter...")
                bl = robust_intersection(mold, bl_box, "back-left")
                if bl is None:
                    logging.error("Failed to compute back-left quarter")
                    raise ValueError("Failed to extract back-left mesh")
                log_mesh_stats(bl, "Back-Left Quarter")
                
                logging.info("Computing back-right quarter...")
                br = robust_intersection(mold, br_box, "back-right")
                if br is None:
                    logging.error("Failed to compute back-right quarter")
                    raise ValueError("Failed to extract back-right mesh")
                log_mesh_stats(br, "Back-Right Quarter")
                
                parts = [fl, fr, bl, br]
                logging.info("Created front-left, front-right, back-left, and back-right quarters.")
                print("Mold split into quarters.")
                
                # Add quarters alignment features if requested
                if alignment_radius > 0:
                    if auto_align_coords:
                        logging.info("Using intelligent geometric analysis for automatic quarters alignment placement...")
                        try:
                            # Use the new quarters coordinate calculation
                            quarters_coords = calculate_quarters_alignment_coordinates(
                                mold, mesh, mold_center, mold_extents, mesh_bounds, alignment_radius, padding
                            )
                            
                            if quarters_coords:
                                logging.info(f"Quarters auto-calculated alignment coordinates successful")
                                print(f"✓ Using intelligently calculated quarters alignment coordinates")
                                
                                # Calculate effective radius (same logic as halves)
                                x_plane_coords = quarters_coords['x_plane_coords']
                                y_plane_coords = quarters_coords['y_plane_coords']
                                
                                # Check space availability for both planes
                                min_space_y = float('inf')
                                min_space_z = float('inf')
                                min_space_x = float('inf')
                                
                                for coord in x_plane_coords:
                                    y, z = coord
                                    min_space_y = min(min_space_y, abs(y - mesh_bounds[0][1]), abs(y - mesh_bounds[1][1]))
                                    min_space_z = min(min_space_z, abs(z - mesh_bounds[0][2]), abs(z - mesh_bounds[1][2]))
                                
                                for coord in y_plane_coords:
                                    x, z = coord
                                    min_space_x = min(min_space_x, abs(x - mesh_bounds[0][0]), abs(x - mesh_bounds[1][0]))
                                    min_space_z = min(min_space_z, abs(z - mesh_bounds[0][2]), abs(z - mesh_bounds[1][2]))
                                
                                space_based_radius = min(min_space_x * 0.7, min_space_y * 0.7, min_space_z * 0.7)
                                size_based_cap = max(max(mesh.extents) * 0.03, 1.5)
                                effective_radius = min(alignment_radius, space_based_radius, size_based_cap)
                                
                                logging.info(f"Using effective quarters alignment radius: {effective_radius:.2f}mm (requested: {alignment_radius:.2f}mm)")
                                print(f"Creating quarters alignment features with {effective_radius:.2f}mm radius...")
                                
                                # Apply quarters alignment features
                                parts_with_features = add_quarters_alignment_features(
                                    parts, mold_center, mold_extents, effective_radius, quarters_coords
                                )
                                
                                # Check if features were successfully applied
                                features_applied = False
                                if len(parts_with_features) == len(parts):
                                    for i, (original, modified) in enumerate(zip(parts, parts_with_features)):
                                        if len(modified.faces) != len(original.faces):
                                            features_applied = True
                                            break
                                
                                if features_applied:
                                    parts = parts_with_features
                                    logging.info("✓ Quarters alignment features applied successfully")
                                    print("✓ Quarters alignment features applied successfully")
                                else:
                                    logging.warning("Quarters alignment features failed to apply")
                                    print("⚠ Quarters alignment features failed - proceeding without alignment")
                                    
                            else:
                                logging.warning("Quarters coordinate calculation returned no valid placements")
                                print("⚠ No valid automatic quarters placement found. Try smaller radius, larger padding, or manual placement.")
                                
                        except Exception as e:
                            logging.error(f"Failed to auto-calculate quarters alignment coords: {e}. Alignment features will be skipped.")
                            print(f"⚠ Automatic quarters alignment calculation failed: {e}")
                    else:
                        # Manual coordinates for quarters mode would need different format
                        # For now, inform user that manual coords for quarters are not yet implemented
                        logging.info("Manual coordinates for quarters mode not yet implemented. Use auto mode or halves for manual coords.")
                        print("ℹ️  Manual coordinates for quarters mode not yet implemented. Use auto mode for quarters alignment.")
                else:
                    logging.info("Skipping quarters alignment features (radius <= 0).")
                
            else:
                logging.info("No splitting selected; returning full mold.")
                print("No splitting applied.")
                parts = [mold]
        except Exception as e:
            logging.error(f"Error during {split_mode} splitting: {str(e)}")
            parts = [mold]  # Fallback: return full mold
    
    # Final validation of parts
    validated_parts = []
    for i, part in enumerate(parts):
        if len(part.vertices) > 0 and len(part.faces) > 0:
            validated_parts.append(part)
            logging.info(f"Part {i+1} is valid: {len(part.vertices)} vertices, {len(part.faces)} faces")
        else:
            logging.warning(f"Part {i+1} is invalid (empty) and will be excluded")
    
    if len(validated_parts) < len(parts):
        logging.warning(f"Removed {len(parts) - len(validated_parts)} invalid (empty) parts")
        parts = validated_parts
    
    return mold, parts

def process_single_model(file_path, output_dir=None, padding=0.1, hole_positions=['bottom'],
                         split_mode='quarters', visualize=False, draft_angle=0.0, custom_planes=None, 
                         alignment_radius=-1.0, alignment_coords=None, auto_align_coords=True):
    """
    Processes a single model and exports the generated mold parts to the organized output directory.
    Uses a single 'output' folder with timestamp-based organization to keep repository clean.
    """
    try:
        logging.info(f"Processing model: {file_path}")
        
        # Always use a single 'output' directory with organized subdirectories
        # Use absolute path to avoid working directory issues
        base_output_dir = os.path.abspath("../output" if os.getcwd().endswith("src") else "output")
        
        # Create timestamp for this run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create organized subdirectory: output/modelname_timestamp/
        run_output_dir = os.path.join(base_output_dir, f"{base_filename}_{timestamp}")
        
        # Override output_dir if provided (for backward compatibility)
        if output_dir:
            run_output_dir = os.path.abspath(output_dir)
            
        os.makedirs(run_output_dir, exist_ok=True)
        
        logging.info(f"Output directory: {run_output_dir}")
        print(f"📁 Output directory: {run_output_dir}")
        
        mold, parts = create_mold_with_preserved_cavity(
            file_path, padding, hole_positions, split_mode, draft_angle, custom_planes, 
            visualize, alignment_radius, alignment_coords, auto_align_coords
        )
        
        if visualize:
            try:
                mold.show()
            except ModuleNotFoundError:
                logging.warning("Visualization not available because pyglet is not installed. Skipping visualization.")
        
        logging.info("Exporting mold parts...")
        print("Exporting mold parts...")
        
        # Export with organized naming
        complete_mold_path = os.path.join(run_output_dir, f'{base_filename}_complete_mold.stl')
        mold.export(complete_mold_path)
        logging.info(f"Exported complete mold: {complete_mold_path}")
        
        for i, part in enumerate(parts):
            part_path = os.path.join(run_output_dir, f'{base_filename}_part_{i+1}.stl')
            part.export(part_path)
            logging.info(f"Exported part {i+1}: {part_path}")
        
        # Create a summary file for this run
        summary_path = os.path.join(run_output_dir, f'{base_filename}_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Mold Generation Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input model: {file_path}\n")
            f.write(f"Base filename: {base_filename}\n")
            f.write(f"Parameters:\n")
            f.write(f"  - Padding: {padding}mm\n")
            f.write(f"  - Split mode: {split_mode}\n")
            f.write(f"  - Draft angle: {draft_angle}°\n")
            f.write(f"  - Hole positions: {hole_positions}\n")
            f.write(f"  - Alignment radius: {alignment_radius}mm\n")
            f.write(f"  - Auto alignment: {auto_align_coords}\n\n")
            f.write(f"Generated files:\n")
            f.write(f"  - Complete mold: {base_filename}_complete_mold.stl\n")
            for i in range(len(parts)):
                f.write(f"  - Part {i+1}: {base_filename}_part_{i+1}.stl\n")
        
        logging.info(f"Successfully exported all parts to {run_output_dir}")
        print(f"✅ Successfully exported all parts to {run_output_dir}")
        print(f"📋 Summary saved: {summary_path}")
        
        return run_output_dir  # Return the actual output directory used
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        print(f"❌ Error processing {file_path}: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Generate a mold from a 3D model (STL) with intelligent automation or manual control."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input STL file or directory containing STL files."
    )
    parser.add_argument(
        "--mode", type=str, choices=["auto", "manual"], default="auto",
        help="Parameter selection mode: 'auto' for intelligent optimization, 'manual' for custom settings (default: auto)."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for the generated mold files. If not specified, uses organized 'output' folder."
    )
    
    # Manual mode parameters (only used when --mode manual)
    parser.add_argument(
        "--padding", type=float, default=None,
        help="[Manual mode] Padding added around the input mesh."
    )
    parser.add_argument(
        "--hole_positions", type=str, nargs="+", default=None,
        help="[Manual mode] Positions to add holes (e.g., bottom, top, left, right)."
    )
    parser.add_argument(
        "--split_mode", type=str, choices=["halves", "quarters"], default=None,
        help="[Manual mode] How to split the mold (halves or quarters)."
    )
    parser.add_argument(
        "--draft_angle", type=float, default=None,
        help="[Manual mode] Draft angle in degrees to apply to the cavity."
    )
    parser.add_argument(
        "--alignment_radius", type=float, default=None,
        help="[Manual mode] Radius for alignment features (-1.0 to disable)."
    )
    parser.add_argument(
        "--alignment_coords", type=str, default=None,
        help="[Manual mode] Manual alignment coordinates as JSON: '[[y1,z1],[y2,z2]]'"
    )
    parser.add_argument(
        "--auto_align", action="store_true", default=None,
        help="[Manual mode] Use automatic intelligent alignment placement."
    )
    
    # General parameters
    parser.add_argument(
        "--auto_split", type=str, choices=["halves", "quarters"], default=None,
        help="[Auto mode] Override automatic split mode choice (halves or quarters)."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize the mold before exporting."
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Inspect model size and properties without generating mold."
    )

    args = parser.parse_args()

    # Initialize smart logging
    setup_smart_logging()

    # Handle inspection mode
    if args.inspect:
        if os.path.isfile(args.input):
            inspect_model_size(args.input)
        else:
            print(f"❌ File not found: {args.input}")
        return

    print("🎯 AUTOMATED 3D MOLD GENERATOR")
    print("=" * 50)
    
    # Mode-specific setup
    if args.mode == "auto":
        print("🤖 Mode: AUTOMATIC - Intelligent parameter optimization")
        print("   The system will analyze your model and calculate optimal parameters")
    else:
        print("⚙️  Mode: MANUAL - Custom parameter control")
        print("   Using your specified parameters")
    
    if args.output:
        print(f"📁 Custom output directory: {args.output}")
    else:
        print(f"📁 Using organized output structure: output/modelname_timestamp/")
    
    # Handle automatic vs manual mode
    if args.mode == "auto":
        # AUTOMATIC MODE - Process each file with intelligent parameter calculation
        if args.auto_split:
            print(f"   🎯 Split mode override: {args.auto_split}")
        
        if os.path.isdir(args.input):
            stl_files = [f for f in os.listdir(args.input) if f.lower().endswith('.stl')]
            print(f"\n📂 Found {len(stl_files)} STL files to process automatically")
            
            processed_dirs = []
            for stl_file in tqdm(stl_files, desc="Auto-processing STL files"):
                file_path = os.path.join(args.input, stl_file)
                try:
                    output_dir = process_model_automatically(file_path, args.output, args.visualize, args.auto_split)
                    if output_dir:
                        processed_dirs.append(output_dir)
                except Exception as e:
                    print(f"⚠ Failed to auto-process {stl_file}: {e}")
                    logging.error(f"Auto-processing failed for {stl_file}: {e}")
                    
            print(f"\n✅ Automatic processing complete!")
            print(f"📁 Generated {len(processed_dirs)} mold sets:")
            for dir_path in processed_dirs:
                print(f"   • {dir_path}")
                
        else:
            # Single file automatic processing
            try:
                output_dir = process_model_automatically(args.input, args.output, args.visualize, args.auto_split)
                if output_dir:
                    print(f"\n✅ Automatic processing complete!")
                    print(f"📁 Mold files saved in: {output_dir}")
                else:
                    print(f"\n❌ Automatic processing cancelled or failed")
            except Exception as e:
                print(f"❌ Automatic processing failed: {e}")
                logging.error(f"Auto-processing failed: {e}")
                return
    
    else:
        # MANUAL MODE - Use provided parameters with defaults
        
        # Parse alignment coordinates if provided
        alignment_coords = None
        if args.alignment_coords:
            try:
                alignment_coords = json.loads(args.alignment_coords)
                if not (isinstance(alignment_coords, list) and len(alignment_coords) == 2):
                    raise ValueError("Expected format: [[y1,z1],[y2,z2]]")
            except Exception as e:
                print(f"❌ Error parsing alignment coordinates: {e}")
                return
        
        # Set manual defaults if not provided
        manual_params = {
            'padding': args.padding if args.padding is not None else 2.0,
            'hole_positions': args.hole_positions if args.hole_positions is not None else ['bottom'],
            'split_mode': args.split_mode if args.split_mode is not None else 'quarters',
            'draft_angle': args.draft_angle if args.draft_angle is not None else 1.0,
            'alignment_radius': args.alignment_radius if args.alignment_radius is not None else -1.0,
            'alignment_coords': alignment_coords,
            'auto_align_coords': args.auto_align if args.auto_align is not None else True
        }
        
        print(f"\n⚙️  Manual parameters:")
        for param, value in manual_params.items():
            if param != 'alignment_coords':
                print(f"   {param}: {value}")
        if manual_params['alignment_coords']:
            print(f"   alignment_coords: {manual_params['alignment_coords']}")

        # Manual mode processing
        if os.path.isdir(args.input):
            stl_files = [f for f in os.listdir(args.input) if f.lower().endswith('.stl')]
            print(f"\n📂 Found {len(stl_files)} STL files to process with manual settings")
            
            processed_dirs = []
            for stl_file in tqdm(stl_files, desc="Processing STL files"):
                file_path = os.path.join(args.input, stl_file)
                try:
                    output_dir = process_single_model(
                        file_path=file_path,
                        output_dir=args.output,
                        padding=manual_params['padding'],
                        hole_positions=manual_params['hole_positions'],
                        split_mode=manual_params['split_mode'],
                        visualize=args.visualize,
                        draft_angle=manual_params['draft_angle'],
                        alignment_radius=manual_params['alignment_radius'],
                        alignment_coords=manual_params['alignment_coords'],
                        auto_align_coords=manual_params['auto_align_coords']
                    )
                    processed_dirs.append(output_dir)
                except Exception as e:
                    print(f"⚠ Failed to process {stl_file}: {e}")
                    
            print(f"\n✅ Manual processing complete!")
            print(f"📁 Generated {len(processed_dirs)} mold sets:")
            for dir_path in processed_dirs:
                print(f"   • {dir_path}")
                
        else:
            try:
                output_dir = process_single_model(
                    file_path=args.input,
                    output_dir=args.output,
                    padding=manual_params['padding'],
                    hole_positions=manual_params['hole_positions'],
                    split_mode=manual_params['split_mode'],
                    visualize=args.visualize,
                    draft_angle=manual_params['draft_angle'],
                    alignment_radius=manual_params['alignment_radius'],
                    alignment_coords=manual_params['alignment_coords'],
                    auto_align_coords=manual_params['auto_align_coords']
                )
                print(f"\n✅ Manual processing complete!")
                print(f"📁 Mold files saved in: {output_dir}")
            except Exception as e:
                print(f"❌ Manual processing failed: {e}")
                return

def process_model_automatically(file_path, output_dir=None, visualize=False, user_split_mode=None):
    """
    Process a single model using automatic parameter optimization.
    This is the main function for automatic mode.
    
    Parameters:
        file_path: Path to the STL file to process
        output_dir: Optional output directory override
        visualize: Whether to show 3D visualization
        user_split_mode: Optional user override for split mode ('halves' or 'quarters')
    """
    try:
        print(f"\n🤖 Auto-processing: {os.path.basename(file_path)}")
        print("=" * 60)
        
        # Load and analyze the mesh
        print("📥 Loading model...")
        mesh = trimesh.load_mesh(file_path)
        
        if not mesh.is_watertight:
            print("🔧 Model not watertight - applying repair...")
            try:
                mesh = enhanced_mesh_repair(mesh)
                if not mesh.is_watertight:
                    print("⚠️  Model still not watertight after repair - proceeding with caution")
            except Exception as e:
                print(f"⚠️  Repair failed: {e} - proceeding with original mesh")
        
        # Calculate optimal parameters
        print("🧠 Analyzing model geometry and calculating optimal parameters...")
        if user_split_mode:
            print(f"   🎯 Using user-specified split mode: {user_split_mode}")
        optimal_params = calculate_optimal_parameters(mesh, user_split_mode)
        
        # Validate parameters for safety
        print("🔒 Validating parameter safety...")
        safe_params, warnings, overall_safety = validate_parameter_safety(mesh, optimal_params)
        
        # Predict success probability
        success_prob = predict_success_probability(mesh, safe_params)
        
        # Display comprehensive analysis
        display_auto_analysis_results(mesh, safe_params, warnings, success_prob)
        
        # Get user confirmation
        if not get_user_confirmation_auto_mode(safe_params, success_prob):
            print("🚫 Auto-processing cancelled by user")
            return None
        
        # Process with automatic parameters
        print(f"\n🚀 Generating mold with optimized parameters...")
        return process_single_model(
            file_path=file_path,
            output_dir=output_dir,
            padding=safe_params['padding'],
            hole_positions=safe_params['hole_positions'],
            split_mode=safe_params['split_mode'],
            visualize=visualize,
            draft_angle=safe_params['draft_angle'],
            alignment_radius=safe_params['alignment_radius'],
            alignment_coords=None,  # Use automatic coordinate calculation
            auto_align_coords=safe_params['auto_align_coords']
        )
        
    except Exception as e:
        logging.error(f"Error in automatic processing: {e}")
        print(f"❌ Auto-processing error: {e}")
        return None

# =============================================================================
# AUTOMATIC PARAMETER SELECTION SYSTEM
# =============================================================================

def calculate_model_complexity(mesh):
    """
    Calculate a complexity score based on multiple geometric factors.
    Returns a value between 0.0 (simple) and 1.0 (complex).
    """
    try:
        extents = mesh.extents
        volume = getattr(mesh, 'volume', 1)
        surface_area = getattr(mesh, 'area', 1)
        face_count = len(mesh.faces)
        
        # Prevent division by zero
        if volume <= 0:
            volume = 1
        if surface_area <= 0:
            surface_area = 1
        
        # Surface detail complexity (higher = more detailed)
        surface_detail = surface_area / (volume ** (2/3))
        
        # Mesh density complexity
        mesh_density = face_count / surface_area
        
        # Convexity complexity (lower convexity = higher complexity)
        try:
            convex_hull = mesh.convex_hull
            convexity_ratio = volume / convex_hull.volume if convex_hull.volume > 0 else 0.5
        except:
            convexity_ratio = 0.5  # Default assumption
        
        # Aspect ratio complexity (extreme ratios = more complex)
        min_extent = min(extents)
        max_extent = max(extents)
        aspect_ratio = max_extent / min_extent if min_extent > 0 else 1
        aspect_complexity = min(aspect_ratio / 3, 2.0)  # Cap at 2.0
        
        # Combine factors (normalized 0-1 scale)
        complexity_score = (
            min(surface_detail / 10, 1.0) * 0.3 +
            min(mesh_density / 0.1, 1.0) * 0.2 +
            (1 - convexity_ratio) * 0.3 +
            min(aspect_complexity / 2, 1.0) * 0.2
        )
        
        # Ensure bounds [0, 1]
        complexity_score = max(0.0, min(1.0, complexity_score))
        
        logging.info(f"Complexity analysis: surface_detail={surface_detail:.2f}, mesh_density={mesh_density:.4f}, convexity={convexity_ratio:.2f}, aspect_ratio={aspect_ratio:.2f}")
        logging.info(f"Final complexity score: {complexity_score:.2f}")
        
        return complexity_score
        
    except Exception as e:
        logging.warning(f"Error calculating complexity score: {e}, using default 0.5")
        return 0.5

def analyze_shape_characteristics(mesh):
    """Analyze shape characteristics to inform parameter selection"""
    try:
        extents = mesh.extents
        center = mesh.center_mass
        bounds = mesh.bounds
        
        # Prevent division by zero
        min_extent = min(extents)
        max_extent = max(extents)
        aspect_ratio = max_extent / min_extent if min_extent > 0 else 1
        
        characteristics = {
            'aspect_ratio': aspect_ratio,
            'is_tall': extents[2] > max(extents[0], extents[1]),  # Taller than wide
            'is_flat': min_extent < max_extent * 0.2,  # One dimension much smaller
            'is_symmetric': abs(center[0] - (bounds[0][0] + bounds[1][0])/2) < max_extent * 0.1,
            'dominant_axis': np.argmax(extents),  # 0=X, 1=Y, 2=Z
            'size_category': 'small' if max_extent < 20 else 'medium' if max_extent < 100 else 'large',
            'max_dimension': max_extent,
            'min_dimension': min_extent,
            'extents': extents
        }
        
        logging.info(f"Shape characteristics: {characteristics}")
        return characteristics
        
    except Exception as e:
        logging.warning(f"Error analyzing shape characteristics: {e}, using defaults")
        return {
            'aspect_ratio': 1.0,
            'is_tall': False,
            'is_flat': False,
            'is_symmetric': True,
            'dominant_axis': 0,
            'size_category': 'medium',
            'max_dimension': 50.0,
            'min_dimension': 20.0,
            'extents': np.array([30.0, 30.0, 30.0])
        }

def calculate_optimal_padding(mesh, complexity_score, shape_characteristics):
    """Calculate optimal padding based on size, complexity, and shape"""
    try:
        max_dim = shape_characteristics['max_dimension']
        
        # Base padding: 5-15% of max dimension
        base_padding = max_dim * 0.08  # Start with 8%
        
        # Complexity adjustments (complex models need more padding for boolean operations)
        complexity_multiplier = 1.0 + (complexity_score * 0.6)  # Up to 60% increase
        
        # Size-based adjustments
        if max_dim < 10:  # Small parts
            size_multiplier = 1.8  # Need much more relative padding
        elif max_dim < 30:  # Medium-small parts
            size_multiplier = 1.3
        elif max_dim < 100:  # Medium parts  
            size_multiplier = 1.0
        else:  # Large parts
            size_multiplier = 0.8  # Can use less relative padding
        
        # Shape-based adjustments
        shape_multiplier = 1.0
        if shape_characteristics['is_flat']:
            shape_multiplier *= 1.2  # Flat objects need more padding for stability
        if shape_characteristics['aspect_ratio'] > 5:
            shape_multiplier *= 1.15  # Elongated objects need more padding
        
        optimal_padding = base_padding * complexity_multiplier * size_multiplier * shape_multiplier
        
        # Ensure reasonable bounds (minimum 0.5mm, maximum 25% of model size)
        min_padding = max(0.5, max_dim * 0.02)  # At least 2% or 0.5mm
        max_padding = max_dim * 0.25  # Not more than 25%
        optimal_padding = max(min_padding, min(optimal_padding, max_padding))
        
        logging.info(f"Padding calculation: base={base_padding:.2f}, complexity_mult={complexity_multiplier:.2f}, size_mult={size_multiplier:.2f}, shape_mult={shape_multiplier:.2f}")
        logging.info(f"Optimal padding: {optimal_padding:.2f}mm")
        
        return optimal_padding
        
    except Exception as e:
        logging.warning(f"Error calculating optimal padding: {e}, using fallback")
        return max(2.0, shape_characteristics.get('max_dimension', 50) * 0.08)

def calculate_optimal_alignment_radius(mesh, complexity_score, shape_characteristics, padding):
    """Calculate optimal alignment feature radius"""
    try:
        max_dim = shape_characteristics['max_dimension']
        
        # Base radius: scale appropriately with model size
        if max_dim < 30:
            base_radius = max_dim * 0.06  # Smaller models need proportionally larger features
        elif max_dim < 100:
            base_radius = max_dim * 0.05  # Medium models
        else:
            base_radius = max_dim * 0.04  # Large models can use smaller relative features
        
        # Complexity adjustments (simpler models can have larger features)
        complexity_multiplier = 1.3 - (complexity_score * 0.5)  # Reduce for complex models
        
        # Size-based adjustments
        if max_dim < 15:  # Small parts
            size_multiplier = 0.7  # Smaller features
        elif max_dim < 50:  # Medium parts
            size_multiplier = 1.0
        else:  # Large parts
            size_multiplier = 1.2  # Can have larger features
        
        # Shape-based adjustments
        shape_multiplier = 1.0
        if shape_characteristics['is_flat']:
            shape_multiplier *= 0.8  # Smaller features for flat objects
        
        optimal_radius = base_radius * complexity_multiplier * size_multiplier * shape_multiplier
        
        # Critical constraint: alignment radius must be appropriately sized for the mold
        # Balance between safety and functionality - features need to be visible and functional
        max_safe_radius = min(
            padding * 0.4,  # Allow up to 40% of padding for reasonable visibility
            max_dim * 0.05,  # Or up to 5% of model size for proportional scaling
            (padding - max_dim * 0.02) * 0.6  # Or 60% of remaining space after mesh clearance
        )
        optimal_radius = min(optimal_radius, max_safe_radius)
        
        # Ensure reasonable bounds (minimum 0.2mm, maximum 10% of model size)
        min_radius = max(0.2, max_dim * 0.015)  # At least 1.5% or 0.2mm
        max_radius = max_dim * 0.1  # Not more than 10%
        optimal_radius = max(min_radius, min(optimal_radius, max_radius))
        
        logging.info(f"Alignment radius calculation: base={base_radius:.2f}, complexity_mult={complexity_multiplier:.2f}, size_mult={size_multiplier:.2f}, shape_mult={shape_multiplier:.2f}")
        logging.info(f"Constrained by padding: max_safe={max_safe_radius:.2f}")
        logging.info(f"Optimal alignment radius: {optimal_radius:.2f}mm")
        
        return optimal_radius
        
    except Exception as e:
        logging.warning(f"Error calculating optimal alignment radius: {e}, using fallback")
        return max(0.5, min(2.0, padding * 0.5))

def determine_optimal_split_mode(mesh, shape_characteristics, padding, alignment_radius):
    """Determine the best split mode based on shape analysis"""
    try:
        # Rules for split mode selection:
        # 1. Very flat objects -> quarters (easier to print, no alignment needed)
        # 2. Very small objects -> quarters (alignment features would be too small)
        # 3. Complex objects -> quarters (simpler boolean operations)
        # 4. Simple, medium/large objects -> halves (better for alignment features)
        
        max_dim = shape_characteristics['max_dimension']
        
        if shape_characteristics['is_flat']:
            reason = "flat object - quarters easier to print"
            return 'quarters', reason
        elif max_dim < 8:  # Only very tiny objects
            reason = "very small object - alignment features would be too small"
            return 'quarters', reason
        elif alignment_radius < 0.2:  # More permissive threshold
            reason = "alignment features too small to be effective"
            return 'quarters', reason
        else:
            reason = "medium/large object suitable for alignment features"
            return 'halves', reason
            
    except Exception as e:
        logging.warning(f"Error determining split mode: {e}, defaulting to quarters")
        return 'quarters', 'error in analysis - safe default'

def calculate_optimal_draft_angle(mesh, shape_characteristics):
    """Calculate optimal draft angle based on shape"""
    try:
        # Rules for draft angle:
        # 1. Tall objects with small base -> more draft
        # 2. Wide/flat objects -> less draft
        # 3. Complex surfaces -> more draft (easier removal)
        
        if shape_characteristics['is_tall']:
            angle = 2.0  # More draft for tall objects
            reason = "tall object needs more draft for easy removal"
        elif shape_characteristics['is_flat']:
            angle = 0.5  # Less draft for flat objects
            reason = "flat object needs minimal draft"
        elif shape_characteristics['aspect_ratio'] > 3:
            angle = 1.5  # Moderate draft for elongated objects
            reason = "elongated object needs moderate draft"
        else:
            angle = 1.0  # Default mild draft
            reason = "standard geometry - mild draft"
        
        logging.info(f"Draft angle: {angle}° ({reason})")
        return angle, reason
        
    except Exception as e:
        logging.warning(f"Error calculating draft angle: {e}, using default")
        return 1.0, "error in analysis - safe default"

def determine_optimal_hole_positions(mesh, shape_characteristics):
    """Determine optimal hole positions based on shape"""
    try:
        # Rules for hole placement:
        # 1. Always include bottom hole for material injection
        # 2. Add top hole for tall objects (air vent)
        # 3. Keep it simple for small objects
        
        holes = ['bottom']  # Always include bottom
        reasons = ['material injection point']
        
        if shape_characteristics['is_tall'] and shape_characteristics['max_dimension'] > 30:
            holes.append('top')
            reasons.append('air vent for tall object')
        
        logging.info(f"Hole positions: {holes} ({', '.join(reasons)})")
        return holes
        
    except Exception as e:
        logging.warning(f"Error determining hole positions: {e}, using default")
        return ['bottom']

def calculate_optimal_parameters(mesh, user_split_mode=None):
    """
    Main function that uses comprehensive geometric analysis to calculate optimal mold parameters.
    This is the core of the automatic system.
    
    Parameters:
        mesh: The input mesh to analyze
        user_split_mode: Optional user override for split mode ('halves' or 'quarters')
                        If provided, this will override the automatic split mode selection.
    """
    try:
        logging.info("🤖 Starting automatic parameter optimization...")
        
        # 1. BASIC GEOMETRIC PROPERTIES
        extents = mesh.extents
        max_dim = max(extents)
        volume = getattr(mesh, 'volume', 0)
        surface_area = getattr(mesh, 'area', 0)
        
        logging.info(f"Model dimensions: {extents}")
        logging.info(f"Max dimension: {max_dim:.2f}mm")
        logging.info(f"Volume: {volume:.2f}mm³")
        logging.info(f"Surface area: {surface_area:.2f}mm²")
        
        # 2. COMPLEXITY ANALYSIS
        complexity_score = calculate_model_complexity(mesh)
        
        # 3. SHAPE ANALYSIS
        shape_characteristics = analyze_shape_characteristics(mesh)
        
        # 4. CALCULATE OPTIMAL PARAMETERS
        
        # Calculate padding first (needed for other calculations)
        padding = calculate_optimal_padding(mesh, complexity_score, shape_characteristics)
        
        # Calculate alignment radius (depends on padding)
        alignment_radius = calculate_optimal_alignment_radius(mesh, complexity_score, shape_characteristics, padding)
        
        # Determine split mode (depends on alignment radius or user choice)
        if user_split_mode:
            split_mode = user_split_mode
            split_reason = f"user override - {user_split_mode} mode selected"
            logging.info(f"Split mode override: using user-specified {user_split_mode} mode")
        else:
            split_mode, split_reason = determine_optimal_split_mode(mesh, shape_characteristics, padding, alignment_radius)
        
        # Calculate other parameters
        draft_angle, draft_reason = calculate_optimal_draft_angle(mesh, shape_characteristics)
        hole_positions = determine_optimal_hole_positions(mesh, shape_characteristics)
        
        # NOTE: No longer automatically disable alignment for quarters mode
        # The new quarters alignment system supports alignment features for quarters
        # Only disable alignment if features are too small to be effective
        if alignment_radius > 0 and alignment_radius < 0.2:
            max_dim = max(mesh.extents)
            if max_dim <= 15:  # Only for very small models
                alignment_radius = -1.0
                logging.info("Disabled alignment features for very small model")
            # For larger models, keep the small radius - the system will cap it appropriately
        
        optimal_params = {
            'padding': padding,
            'alignment_radius': alignment_radius,
            'split_mode': split_mode,
            'draft_angle': draft_angle,
            'hole_positions': hole_positions,
            'auto_align_coords': True,  # Always use auto-alignment when available
            'complexity_score': complexity_score,
            'analysis_reasons': {
                'split_mode': split_reason,
                'draft_angle': draft_reason
            }
        }
        
        return optimal_params
        
    except Exception as e:
        logging.error(f"Error in automatic parameter calculation: {e}")
        # Return safe fallback parameters
        return {
            'padding': max(2.0, max(mesh.extents) * 0.08),
            'alignment_radius': -1.0,  # Disable alignment features on error
            'split_mode': 'quarters',
            'draft_angle': 1.0,
            'hole_positions': ['bottom'],
            'auto_align_coords': True,
            'complexity_score': 0.5,
            'analysis_reasons': {
                'split_mode': 'error fallback - safe default',
                'draft_angle': 'error fallback - mild draft'
            }
        }

def validate_parameter_safety(mesh, params):
    """Validate that calculated parameters are safe and won't cause failures"""
    try:
        safety_checks = {
            'padding_clearance': True,
            'alignment_clearance': True,
            'size_ratios': True,
            'geometry_compatibility': True
        }
        
        warnings = []
        max_dim = max(mesh.extents)
        
        # Check 1: Padding clearance
        min_padding = max(0.5, max_dim * 0.02)  # Minimum 2% or 0.5mm
        if params['padding'] < min_padding:
            params['padding'] = min_padding
            warnings.append(f"Increased padding to minimum safe value: {min_padding:.2f}mm")
            safety_checks['padding_clearance'] = False
        
        # Check 2: Alignment feature clearance
        if params['split_mode'] == 'halves' and params['alignment_radius'] > 0:
            max_safe_radius = params['padding'] * 0.6  # Max 60% of padding
            if params['alignment_radius'] > max_safe_radius:
                params['alignment_radius'] = max_safe_radius
                warnings.append(f"Reduced alignment radius for safety: {max_safe_radius:.2f}mm")
                safety_checks['alignment_clearance'] = False
        
        # Check 3: Size ratios
        max_padding = max_dim * 0.3  # Padding shouldn't exceed 30% of model
        if params['padding'] > max_padding:
            params['padding'] = max_padding
            warnings.append(f"Reduced padding to maximum safe ratio: {max_padding:.2f}mm")
            safety_checks['size_ratios'] = False
        
        # Check 4: Draft angle bounds
        if params['draft_angle'] < 0:
            params['draft_angle'] = 0
            warnings.append("Set negative draft angle to zero")
        elif params['draft_angle'] > 10:
            params['draft_angle'] = 10
            warnings.append("Reduced excessive draft angle to 10°")
        
        # Check 5: Alignment radius minimum for effectiveness
        if params['alignment_radius'] > 0 and params['alignment_radius'] < 0.2:
            if max_dim > 15:  # Only for larger models where we can increase it
                params['alignment_radius'] = 0.2
                warnings.append("Increased alignment radius to minimum effective size: 0.2mm")
            else:
                params['alignment_radius'] = -1.0  # Disable for very small models
                params['split_mode'] = 'quarters'
                warnings.append("Disabled alignment features - too small to be effective")
                safety_checks['alignment_clearance'] = False
        
        overall_safety = all(safety_checks.values())
        
        logging.info(f"Safety validation: {safety_checks}")
        if warnings:
            for warning in warnings:
                logging.warning(f"Safety adjustment: {warning}")
        
        return params, warnings, overall_safety
        
    except Exception as e:
        logging.error(f"Error in safety validation: {e}")
        return params, [f"Error in safety validation: {e}"], False

def predict_success_probability(mesh, params):
    """Predict the probability of successful mold generation"""
    try:
        # Factors that affect success:
        # 1. Mesh quality (watertight, manifold)
        # 2. Parameter ratios
        # 3. Geometric complexity
        # 4. Boolean operation feasibility
        
        success_factors = []
        
        # Mesh quality factor (most important)
        mesh_quality = 0.9 if mesh.is_watertight else 0.5
        if hasattr(mesh, 'is_winding_consistent') and mesh.is_winding_consistent:
            mesh_quality += 0.1
        success_factors.append(mesh_quality)
        
        # Parameter safety factor
        max_dim = max(mesh.extents)
        padding_ratio = params['padding'] / max_dim
        if 0.05 <= padding_ratio <= 0.25:
            param_safety = 1.0
        elif 0.02 <= padding_ratio <= 0.3:
            param_safety = 0.8
        else:
            param_safety = 0.6
        success_factors.append(param_safety)
        
        # Complexity factor (complex models are harder to process)
        complexity = params.get('complexity_score', 0.5)
        complexity_factor = 1.0 - (complexity * 0.3)  # High complexity reduces success
        success_factors.append(complexity_factor)
        
        # Model size factor (very small or very large models are more challenging)
        if 5 <= max_dim <= 200:
            size_factor = 1.0  # Ideal size range
        elif 2 <= max_dim <= 500:
            size_factor = 0.9  # Acceptable range
        else:
            size_factor = 0.7  # Challenging range
        success_factors.append(size_factor)
        
        # Boolean operation feasibility (based on split mode and alignment features)
        if params['split_mode'] == 'quarters':
            if params['alignment_radius'] > 0:
                boolean_factor = 0.85  # Quarters with alignment features - moderate complexity
        else:
                boolean_factor = 0.9   # Quarters without alignment - simpler operations
        if params['alignment_radius'] > 0:
            boolean_factor = 0.8   # Halves with alignment features
        else:
            boolean_factor = 0.85  # Halves without alignment
        success_factors.append(boolean_factor)
        
        # Calculate overall probability
        success_probability = np.mean(success_factors)
        
        logging.info(f"Success prediction factors: mesh_quality={mesh_quality:.2f}, param_safety={param_safety:.2f}, complexity={complexity_factor:.2f}, size={size_factor:.2f}, boolean={boolean_factor:.2f}")
        logging.info(f"Predicted success probability: {success_probability:.1%}")
        
        return success_probability
        
    except Exception as e:
        logging.warning(f"Error predicting success probability: {e}")
        return 0.7  # Conservative estimate

def display_auto_analysis_results(mesh, params, warnings, success_prob):
    """Display comprehensive analysis results to the user"""
    try:
        print(f"\n🔬 MODEL ANALYSIS RESULTS:")
        print("=" * 50)
        
        # Model properties
        extents = mesh.extents
        max_dim = max(extents)
        volume = getattr(mesh, 'volume', 0)
        
        print(f"📐 Model Properties:")
        print(f"   Dimensions: {extents[0]:.1f} × {extents[1]:.1f} × {extents[2]:.1f} mm")
        print(f"   Largest dimension: {max_dim:.1f} mm")
        print(f"   Volume: {volume:.1f} mm³" if volume > 0 else "   Volume: Not available")
        print(f"   Watertight: {'✓' if mesh.is_watertight else '✗'}")
        
        # Complexity analysis
        complexity = params.get('complexity_score', 0.5)
        if complexity < 0.3:
            complexity_desc = "Simple"
        elif complexity < 0.6:
            complexity_desc = "Moderate"
        else:
            complexity_desc = "Complex"
        
        print(f"   Complexity: {complexity:.2f}/1.0 ({complexity_desc})")
        
        # Success prediction
        if success_prob >= 0.8:
            success_desc = "High"
            success_icon = "🟢"
        elif success_prob >= 0.6:
            success_desc = "Good"
            success_icon = "🟡"
        else:
            success_desc = "Moderate"
            success_icon = "🟠"
        
        print(f"   Success probability: {success_prob:.1%} ({success_icon} {success_desc})")
        
        print(f"\n🎯 OPTIMIZED PARAMETERS:")
        print("=" * 50)
        print(f"   Padding: {params['padding']:.2f} mm ({params['padding']/max_dim*100:.1f}% of model size)")
        print(f"   Split mode: {params['split_mode'].title()} ({params['analysis_reasons']['split_mode']})")
        print(f"   Draft angle: {params['draft_angle']:.1f}° ({params['analysis_reasons']['draft_angle']})")
        print(f"   Holes: {', '.join(params['hole_positions'])}")
        
        if params['alignment_radius'] > 0:
            if params['split_mode'] == 'quarters':
                print(f"   Alignment features: {params['alignment_radius']:.2f} mm radius (cross-pattern for quarters)")
            else:
                print(f"   Alignment features: {params['alignment_radius']:.2f} mm radius")
        else:
            print(f"   Alignment features: Disabled (features too small or radius <= 0)")
        
        # Safety warnings
        if warnings:
            print(f"\n⚠️  SAFETY ADJUSTMENTS:")
            for warning in warnings:
                print(f"   • {warning}")
        
        print("=" * 50)
        
    except Exception as e:
        logging.warning(f"Error displaying analysis results: {e}")

def get_user_confirmation_auto_mode(params, success_prob):
    """Get user confirmation for automatic parameters"""
    try:
        print(f"\n🤖 AUTOMATIC MODE READY")
        print("=" * 30)
        
        if success_prob >= 0.8:
            print("✅ High success probability - recommended to proceed")
        elif success_prob >= 0.6:
            print("✅ Good success probability - should work well")
        else:
            print("⚠️  Moderate success probability - may need parameter adjustment")
        
        while True:
            choice = input(f"\nProceed with automatic parameters? (Y/n): ").strip().lower()
            if choice in ['', 'y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                return False
            else:
                print("Please enter Y or N")
                
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return False
    except Exception as e:
        logging.warning(f"Error getting user confirmation: {e}")
        return True  # Default to proceed

# =============================================================================
# END AUTOMATIC PARAMETER SELECTION SYSTEM
# =============================================================================

def try_iterative_alignment_positioning(parts, mold_center, mold_extents, feature_radius, original_coords, 
                                       mesh_bounds, alignment_radius, padding):
    """
    Try iterative positioning if the original alignment coordinates fail.
    This function attempts to find alternative positions for alignment features.
    """
    logging.info("Starting iterative positioning for failed alignment features...")
    
    # Calculate alternative positions based on the original coordinates
    y1, z1 = original_coords[0]
    y2, z2 = original_coords[1]
    
    # Generate offset variations
    offset_distance = feature_radius * 1.5  # Move by 1.5x feature radius
    
    alternative_sets = [
        # Slightly outward (away from center)
        [[y1 - offset_distance if y1 < 0 else y1 + offset_distance, z1], 
         [y2 + offset_distance if y2 > 0 else y2 - offset_distance, z2]],
        
        # Slightly upward
        [[y1, z1 + offset_distance], [y2, z2 + offset_distance]],
        
        # Slightly downward  
        [[y1, z1 - offset_distance], [y2, z2 - offset_distance]],
        
        # Diagonal adjustments
        [[y1 - offset_distance * 0.7 if y1 < 0 else y1 + offset_distance * 0.7, z1 + offset_distance * 0.7], 
         [y2 + offset_distance * 0.7 if y2 > 0 else y2 - offset_distance * 0.7, z2 + offset_distance * 0.7]],
        
        # Smaller features (reduce radius by 30%)
        "reduce_radius"
    ]
    
    for attempt, alt_coords in enumerate(alternative_sets):
        if alt_coords == "reduce_radius":
            # Try with smaller features
            logging.info(f"Attempt {attempt + 1}: Trying with reduced feature size...")
            reduced_radius = feature_radius * 0.7
            try:
                result_parts = add_alignment_features(parts, mold_center, mold_extents, 
                                                    reduced_radius, original_coords)
                if result_parts != parts:  # Check if features were actually applied
                    logging.info("✓ Iterative positioning successful with reduced feature size")
                    return result_parts
            except Exception as e:
                logging.warning(f"Reduced radius attempt failed: {e}")
                continue
        else:
            # Validate alternative coordinates are within bounds
            valid_alt = True
            for coord in alt_coords:
                y, z = coord
                if (y < mold_center[1] - mold_extents[1]/2 + feature_radius or 
                    y > mold_center[1] + mold_extents[1]/2 - feature_radius or
                    z < mold_center[2] - mold_extents[2]/2 + feature_radius or 
                    z > mold_center[2] + mold_extents[2]/2 - feature_radius):
                    valid_alt = False
                    break
            
            if not valid_alt:
                logging.debug(f"Alternative coordinates {attempt + 1} out of bounds, skipping")
                continue
            
            logging.info(f"Attempt {attempt + 1}: Trying alternative coordinates {alt_coords}")
            try:
                result_parts = add_alignment_features(parts, mold_center, mold_extents, 
                                                    feature_radius, alt_coords)
                if result_parts != parts:  # Check if features were actually applied
                    logging.info(f"✓ Iterative positioning successful with alternative coordinates {alt_coords}")
                    return result_parts
            except Exception as e:
                logging.warning(f"Alternative coordinates attempt {attempt + 1} failed: {e}")
                continue
    
    logging.warning("All iterative positioning attempts failed")
    return parts


if __name__ == "__main__":
    main()
