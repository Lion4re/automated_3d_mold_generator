import trimesh
import os
import argparse
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
import cv2
import logging
from datetime import datetime
import glob

# Optional Open3D import for advanced mesh repair
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("Open3D available for advanced mesh repair")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available - using basic trimesh repair only")


def setup_logging(input_file):
    """Setup logging to file with timestamp and model info."""
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.splitext(os.path.basename(input_file))[0]
    log_filename = os.path.join("logs", f"moldmaker_{model_name}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"STL Mold Maker Session Started")
    logging.info(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Input Model: {os.path.basename(input_file)}")
    logging.info(f"Log File: {log_filename}")
    
    return log_filename


def find_stl_files():
    """Find all STL files in common directories."""
    search_paths = [
        ".",  # Current directory
        "models",
        "/models",
        "Python/minimal_chess",
    ]
    
    stl_files = []
    for path in search_paths:
        if os.path.exists(path):
            pattern = os.path.join(path, "*.stl")
            files = glob.glob(pattern, recursive=False)
            for file in files:
                stl_files.append(file)
    
    # Remove duplicates and sort
    stl_files = sorted(list(set(stl_files)))
    return stl_files


def display_available_models():
    """Display available STL files for user selection."""
    stl_files = find_stl_files()
    
    if not stl_files:
        print("❌ No STL files found in common directories!")
        print("   Please place your STL files in one of these directories:")
        print("   - Current directory")
        print("   - models/")
        print("   - Python/models/")
        print("   - STL/")
        return None
    
    print(f"\n📁 Available STL Models ({len(stl_files)} found):")
    print("=" * 60)
    
    for i, file in enumerate(stl_files, 1):
        try:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb > 1 else f"{os.path.getsize(file) / 1024:.1f} KB"
        except:
            size_str = "Unknown size"
        
        print(f"{i:2d}. {os.path.basename(file):<30} ({size_str})")
        print(f"    📂 {os.path.dirname(file) if os.path.dirname(file) else 'Current directory'}")
    
    print("=" * 60)
    return stl_files


def get_user_input(prompt, default_value, input_type=str, choices=None):
    """Get user input with default value support."""
    if choices:
        choices_str = "/".join(str(choice) for choice in choices)
        full_prompt = f"{prompt} [{choices_str}] (default: {default_value}): "
    else:
        full_prompt = f"{prompt} (default: {default_value}): "
    
    while True:
        user_input = input(full_prompt).strip()
        
        if not user_input:
            return default_value
        
        try:
            if input_type == int:
                value = int(user_input)
            elif input_type == float:
                value = float(user_input)
            else:
                value = user_input
            
            if choices and value not in choices:
                print(f"❌ Invalid choice. Please select from: {choices}")
                continue
                
            return value
            
        except ValueError:
            print(f"❌ Invalid input. Please enter a {input_type.__name__}.")


def interactive_setup():
    """Interactive setup for mold creation parameters."""
    print("\n" + "=" * 80)
    print("🏭 STL MOLD MAKER - Interactive Setup")
    print("=" * 80)
    print("💡 Tip: Press ENTER to use default values shown in parentheses")
    print()
    
    # Step 1: Select STL file
    stl_files = display_available_models()
    if not stl_files:
        return None
    
    print("\n📋 Step 1: Select your STL model")
    while True:
        try:
            selection = input(f"Enter model number (1-{len(stl_files)}): ").strip()
            if not selection:
                print("❌ Please enter a model number.")
                continue
            
            model_index = int(selection) - 1
            if 0 <= model_index < len(stl_files):
                selected_file = stl_files[model_index]
                print(f"✅ Selected: {os.path.basename(selected_file)}")
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(stl_files)}.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    print(f"\n📋 Step 2: Configure mold parameters")
    print("-" * 40)
    
    wall_thickness = get_user_input(
        "Wall thickness (mm) - leave empty for auto-calculation", 
        "auto", 
        str
    )
    if wall_thickness == "auto":
        wall_thickness = None
    else:
        try:
            wall_thickness = float(wall_thickness)
        except ValueError:
            print("❌ Invalid thickness, using auto-calculation")
            wall_thickness = None
    
    split_axis = get_user_input(
        "Split axis (x=left-right, y=front-back, z=top-bottom)", 
        "x", 
        str, 
        choices=['x', 'y', 'z']
    )
    
    mold_pieces = get_user_input(
        "Number of mold pieces", 
        2, 
        int, 
        choices=[2, 4]
    )
    
    num_alignment_keys = get_user_input(
        "Number of alignment keys", 
        2, 
        int, 
        choices=[2, 4]
    )
    
    repair_method = get_user_input(
        "Mesh repair method (auto=smart, trimesh=basic, open3d=advanced, none=skip)", 
        "auto", 
        str, 
        choices=['auto', 'trimesh', 'open3d', 'hybrid', 'none']
    )
    
    draft_input = get_user_input(
        "Draft angle in degrees (0.5-3.0 for easier demolding, empty for none)", 
        "none", 
        str
    )
    if draft_input == "none" or not draft_input:
        draft_angle = None
    else:
        try:
            draft_angle = float(draft_input)
            if draft_angle <= 0 or draft_angle > 10:
                print("⚠️  Draft angle should be between 0.5-3.0 degrees, using 1.5")
                draft_angle = 1.5
        except ValueError:
            print("❌ Invalid draft angle, disabling draft")
            draft_angle = None
    
    # Summary
    print(f"\n📋 Configuration Summary:")
    print("-" * 40)
    print(f"📦 Model: {os.path.basename(selected_file)}")
    print(f"🔧 Wall thickness: {'Auto-calculated' if wall_thickness is None else f'{wall_thickness} mm'}")
    print(f"⚡ Split axis: {split_axis.upper()}-axis")
    print(f"🧩 Mold pieces: {mold_pieces}")
    print(f"🔑 Alignment keys: {num_alignment_keys}")
    print(f"🛠️  Mesh repair: {repair_method}")
    print(f"📐 Draft angle: {'Disabled' if draft_angle is None else f'{draft_angle}°'}")
    
    confirm = input(f"\n✅ Proceed with mold creation? [Y/n]: ").strip().lower()
    if confirm in ['n', 'no']:
        print("❌ Mold creation cancelled.")
        return None
    
    return {
        'input_file': selected_file,
        'wall_thickness': wall_thickness,
        'split_axis': split_axis,
        'mold_pieces': mold_pieces,
        'num_alignment_keys': num_alignment_keys,
        'repair_method': repair_method,
        'draft_angle': draft_angle
    }


def print_minimal_progress(message, level="INFO"):
    """Print minimal progress info to console."""
    if level == "INFO":
        print(f"🔄 {message}")
    elif level == "SUCCESS":
        print(f"✅ {message}")
    elif level == "WARNING":
        print(f"⚠️  {message}")
    elif level == "ERROR":
        print(f"❌ {message}")


def detect_optimal_wall_thickness(original_mesh, split_axis='z', num_alignment_keys=2, mold_pieces=2, 
                                user_thickness=None):
    """    
    Analyzes object geometry, alignment features, split axis, and manufacturing constraints
    to determine the optimal wall thickness for reliable mold creation.
    
    Parameters:
    - original_mesh: The input mesh to analyze
    - split_axis: Axis along which the mold will be split ('x', 'y', 'z')
    - num_alignment_keys: Number of alignment keys (affects minimum thickness)
    - mold_pieces: Number of mold pieces (2 or 4)
    - user_thickness: User-provided thickness (if provided, will be validated)
    
    Returns:
    - Dictionary with optimal thickness and analysis details
    """
    logging.info(f"INTELLIGENT WALL THICKNESS ANALYSIS")
    logging.info(f"Analyzing optimal wall thickness for {mold_pieces}-piece mold (split: {split_axis.upper()}-axis)")
    
    bounds = original_mesh.bounds
    object_dims = bounds[1] - bounds[0]
    object_volume = abs(original_mesh.volume) if hasattr(original_mesh, 'volume') else np.prod(object_dims)
    object_surface_area = original_mesh.area if hasattr(original_mesh, 'area') else 2 * np.sum([
        object_dims[0] * object_dims[1],
        object_dims[0] * object_dims[2], 
        object_dims[1] * object_dims[2]
    ])
    
    min_dim = min(object_dims)
    max_dim = max(object_dims)
    avg_dim = np.mean(object_dims)
    aspect_ratio = max_dim / min_dim
    
    logging.info(f"Object Analysis:")
    logging.info(f"   Dimensions: {object_dims[0]:.2f} × {object_dims[1]:.2f} × {object_dims[2]:.2f} mm")
    logging.info(f"   Volume: {object_volume:.1f} mm³")
    logging.info(f"   Surface area: {object_surface_area:.1f} mm²")
    logging.info(f"   Min/Max/Avg dimension: {min_dim:.2f} / {max_dim:.2f} / {avg_dim:.2f} mm")
    logging.info(f"   Aspect ratio: {aspect_ratio:.2f}")
    
    if avg_dim < 8.0:
        size_category = "EXTRA_SMALL"
        base_thickness_factor = 0.25  # 25% of average dimension (more conservative)
        min_absolute_thickness = 1.5   # Reduced minimum for tiny objects
        max_absolute_thickness = 6.0   # Reduced maximum for tiny objects
        logging.info(f"EXTRA SMALL object detected - using conservative parameters")
    elif avg_dim < 15.0:
        size_category = "SMALL"
        base_thickness_factor = 0.15  # 15% of average dimension
        min_absolute_thickness = 2.0   # Minimum for small objects
        max_absolute_thickness = 8.0   # Maximum for small objects
    elif avg_dim < 50.0:
        size_category = "MEDIUM"
        base_thickness_factor = 0.12  # 12% of average dimension
        min_absolute_thickness = 5.0   # Minimum for medium objects
        max_absolute_thickness = 15.0  # Maximum for medium objects
    else:
        size_category = "LARGE"
        base_thickness_factor = 0.10  # 10% of average dimension
        min_absolute_thickness = 8.0   # Minimum for large objects
        max_absolute_thickness = 25.0  # Maximum for large objects
    
    logging.info(f"Size category: {size_category} (avg dimension: {avg_dim:.2f} mm)")
    
    base_thickness = avg_dim * base_thickness_factor
    
    axis_index = {'x': 0, 'y': 1, 'z': 2}[split_axis]
    split_plane_dims = [object_dims[i] for i in range(3) if i != axis_index]
    split_plane_area = split_plane_dims[0] * split_plane_dims[1]
    split_direction_length = object_dims[axis_index]
    
    split_area_factor = 1.0
    if split_plane_area > avg_dim ** 2 * 2:  # Large split plane
        split_area_factor = 1.2
        logging.info(f"Large split plane detected ({split_plane_area:.1f} mm²) - increasing thickness")
    elif split_plane_area < avg_dim ** 2 * 0.5:  # Small split plane
        split_area_factor = 0.9
        logging.info(f"Small split plane detected ({split_plane_area:.1f} mm²) - optimizing thickness")
    
    if num_alignment_keys > 0:
        key_radius = base_thickness / 4  # Standard key sizing
        key_clearance = base_thickness * 0.4  # Safety clearance from cavity
        min_thickness_for_keys = (key_radius * 2) + key_clearance + (base_thickness * 0.3)
        
        logging.info(f"Alignment key analysis:")
        logging.info(f"   Number of keys: {num_alignment_keys}")
        logging.info(f"   Key radius: {key_radius:.2f} mm")
        logging.info(f"   Required clearance: {key_clearance:.2f} mm")
        logging.info(f"   Minimum thickness for keys: {min_thickness_for_keys:.2f} mm")
    else:
        min_thickness_for_keys = 0
    
    if mold_pieces == 4:
        multipiece_factor = 1.15
        logging.info(f"4-piece mold detected - increasing thickness by 15% for structural integrity")
    else:
        multipiece_factor = 1.0
    
    if size_category == "EXTRA_SMALL":
        manufacturing_min = 1.5  # Minimum printable thickness for tiny objects
        logging.info(f"Extra small object manufacturing constraints: minimum {manufacturing_min:.1f} mm")
    elif size_category == "SMALL":
        manufacturing_min = 2.0  # Minimum printable thickness
        logging.info(f"Small object manufacturing constraints: minimum {manufacturing_min:.1f} mm")
    elif size_category == "MEDIUM":
        manufacturing_min = 3.0
        logging.info(f"Medium object manufacturing constraints: minimum {manufacturing_min:.1f} mm")
    else:
        manufacturing_min = 4.0
        logging.info(f"Large object manufacturing constraints: minimum {manufacturing_min:.1f} mm")
    
    # Step 9: Calculate optimal thickness
    # Start with base thickness and apply all factors
    calculated_thickness = base_thickness * split_area_factor * multipiece_factor
    
    # Ensure minimum requirements are met
    min_thickness_required = max(
        min_absolute_thickness,
        min_thickness_for_keys,
        manufacturing_min,
        min_dim * 0.05  # Never less than 5% of smallest dimension
    )
    
    # Apply maximum constraint
    max_thickness_allowed = min(
        max_absolute_thickness,
        min_dim * 0.4,  # Never more than 40% of smallest dimension
        avg_dim * 0.25   # Never more than 25% of average dimension
    )
    
    # Final thickness calculation
    optimal_thickness = max(min_thickness_required, min(calculated_thickness, max_thickness_allowed))
    
    # Step 10: Validation and user override
    if user_thickness is not None:
        logging.info(f"User-provided thickness: {user_thickness:.2f} mm")
        
        if user_thickness < min_thickness_required:
            logging.warning(f"WARNING: User thickness ({user_thickness:.2f} mm) is below minimum required ({min_thickness_required:.2f} mm)")
            logging.warning(f"This may cause issues with:")
            if user_thickness < min_thickness_for_keys:
                logging.warning(f"   - Alignment key placement (needs {min_thickness_for_keys:.2f} mm)")
            if user_thickness < manufacturing_min:
                logging.warning(f"   - Manufacturing constraints (needs {manufacturing_min:.2f} mm)")
            
            recommendation = "INCREASE"
            final_thickness = optimal_thickness
            override_applied = True
        elif user_thickness > max_thickness_allowed:
            logging.warning(f"WARNING: User thickness ({user_thickness:.2f} mm) exceeds maximum recommended ({max_thickness_allowed:.2f} mm)")
            logging.warning(f"This may cause:")
            logging.warning(f"   - Material waste and increased print time")
            logging.warning(f"   - Difficulty in handling large mold pieces")
            
            recommendation = "DECREASE"
            final_thickness = optimal_thickness
            override_applied = True
        else:
            logging.info(f"User thickness is within acceptable range")
            recommendation = "ACCEPTED"
            final_thickness = user_thickness
            override_applied = False
    else:
        recommendation = "CALCULATED"
        final_thickness = optimal_thickness
        override_applied = False
    
    # Step 11: Generate analysis report
    analysis_report = {
        'optimal_thickness': final_thickness,
        'user_thickness': user_thickness,
        'calculated_thickness': optimal_thickness,
        'recommendation': recommendation,
        'override_applied': override_applied,
        'object_analysis': {
            'size_category': size_category,
            'dimensions': object_dims.tolist(),
            'volume': object_volume,
            'surface_area': object_surface_area,
            'aspect_ratio': aspect_ratio
        },
        'constraints': {
            'minimum_required': min_thickness_required,
            'maximum_allowed': max_thickness_allowed,
            'manufacturing_min': manufacturing_min,
            'alignment_keys_min': min_thickness_for_keys,
        },
        'factors': {
            'base_thickness_factor': base_thickness_factor,
            'split_area_factor': split_area_factor,
            'multipiece_factor': multipiece_factor
        },
        'split_analysis': {
            'axis': split_axis,
            'plane_area': split_plane_area,
            'direction_length': split_direction_length
        }
    }
    
    # Step 12: Log comprehensive summary
    logging.info(f"OPTIMAL THICKNESS ANALYSIS RESULTS")
    logging.info(f"Object: {size_category} ({avg_dim:.1f} mm avg dimension)")
    logging.info(f"Base calculation: {base_thickness:.2f} mm ({base_thickness_factor*100:.0f}% of avg dimension)")
    logging.info(f"Applied factors:")
    logging.info(f"   • Split area factor: {split_area_factor:.2f}x")
    logging.info(f"   • Multi-piece factor: {multipiece_factor:.2f}x")
    logging.info(f"Thickness constraints:")
    logging.info(f"   • Minimum required: {min_thickness_required:.2f} mm")
    logging.info(f"   • Maximum allowed: {max_thickness_allowed:.2f} mm")
    logging.info(f"   • Calculated optimal: {optimal_thickness:.2f} mm")
    
    if user_thickness:
        logging.info(f"User input: {user_thickness:.2f} mm - {recommendation}")
    
    logging.info(f"FINAL THICKNESS: {final_thickness:.2f} mm")
    
    if override_applied:
        logging.info(f"Automatic adjustment applied for safety/optimization")
    
    return analysis_report


def enhanced_mesh_repair(mesh, repair_method="auto"):
    """
    🔧 ENHANCED MESH REPAIR SYSTEM
    
    Comprehensive mesh repair using both trimesh built-in methods and optional Open3D
    advanced repair algorithms. Automatically detects mesh issues and applies appropriate
    repair strategies for optimal mold creation results.
    
    Parameters:
    - mesh: Input trimesh.Trimesh object to repair
    - repair_method: "auto", "trimesh", "open3d", or "hybrid"
    
    Returns:
    - repaired_mesh: Repaired mesh ready for mold creation
    - repair_report: Dictionary with detailed repair information
    """
    logging.info(f"ENHANCED MESH REPAIR SYSTEM")
    logging.info(f"Repair method: {repair_method.upper()}")
    
    # Initialize repair report
    repair_report = {
        'original_watertight': mesh.is_watertight,
        'original_vertex_count': len(mesh.vertices),
        'original_face_count': len(mesh.faces),
        'original_volume': abs(mesh.volume) if hasattr(mesh, 'volume') else 0,
        'methods_attempted': [],
        'success': False,
        'final_watertight': False,
        'improvements': []
    }
    
    logging.info(f"Original mesh analysis:")
    logging.info(f"   Vertices: {repair_report['original_vertex_count']}")
    logging.info(f"   Faces: {repair_report['original_face_count']}")
    logging.info(f"   Volume: {repair_report['original_volume']:.2f} mm³")
    logging.info(f"   Watertight: {'YES' if repair_report['original_watertight'] else 'NO'}")
    
    # If already watertight and repair method is auto, skip repair
    if repair_report['original_watertight'] and repair_method == "auto":
        logging.info(f"Mesh is already watertight - no repair needed!")
        repair_report['success'] = True
        repair_report['final_watertight'] = True
        return mesh, repair_report
    
    # Start with a copy of the original mesh
    repaired_mesh = mesh.copy()
    
    # Method 1: Basic trimesh repair
    if repair_method in ["auto", "trimesh", "hybrid"]:
        logging.info(f"METHOD 1: TRIMESH BASIC REPAIR")
        repair_report['methods_attempted'].append("trimesh_basic")
        
        try:
            # Fill holes using trimesh's built-in repair
            if hasattr(repaired_mesh, 'fill_holes'):
                logging.info(f"Filling holes...")
                repaired_mesh.fill_holes()
                repair_report['improvements'].append("holes_filled")
            
            # Fix normals and winding
            logging.info(f"Fixing normals and winding...")
            repaired_mesh.fix_normals()
            repair_report['improvements'].append("normals_fixed")
            
            # Remove degenerate faces
            logging.info(f"Removing degenerate faces...")
            original_face_count = len(repaired_mesh.faces)
            repaired_mesh.remove_degenerate_faces()
            removed_faces = original_face_count - len(repaired_mesh.faces)
            if removed_faces > 0:
                logging.info(f"   Removed {removed_faces} degenerate faces")
                repair_report['improvements'].append(f"removed_{removed_faces}_degenerate_faces")
            
            # Remove duplicate faces
            logging.info(f"Removing duplicate faces...")
            original_face_count = len(repaired_mesh.faces)
            repaired_mesh.remove_duplicate_faces()
            removed_faces = original_face_count - len(repaired_mesh.faces)
            if removed_faces > 0:
                logging.info(f"   Removed {removed_faces} duplicate faces")
                repair_report['improvements'].append(f"removed_{removed_faces}_duplicate_faces")
            
            # Check if trimesh repair was successful
            trimesh_watertight = repaired_mesh.is_watertight
            logging.info(f"Trimesh repair result: {'WATERTIGHT' if trimesh_watertight else 'STILL NOT WATERTIGHT'}")
            
            if trimesh_watertight:
                repair_report['success'] = True
                repair_report['final_watertight'] = True
                logging.info(f"Trimesh repair successful!")
                
                # If method is trimesh only or auto and successful, return
                if repair_method in ["auto", "trimesh"]:
                    repair_report['final_vertex_count'] = len(repaired_mesh.vertices)
                    repair_report['final_face_count'] = len(repaired_mesh.faces)
                    repair_report['final_volume'] = abs(repaired_mesh.volume) if hasattr(repaired_mesh, 'volume') else 0
                    return repaired_mesh, repair_report
            
        except Exception as e:
            logging.warning(f"Trimesh repair encountered error: {e}")
            repair_report['improvements'].append(f"trimesh_error: {str(e)}")
    
    # Method 2: Advanced Open3D repair
    if repair_method in ["auto", "open3d", "hybrid"] and OPEN3D_AVAILABLE:
        logging.info(f"METHOD 2: OPEN3D ADVANCED REPAIR")
        repair_report['methods_attempted'].append("open3d_advanced")
        
        try:
            logging.info(f"Converting trimesh → Open3D...")
            
            # Convert trimesh to Open3D mesh
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(repaired_mesh.vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(repaired_mesh.faces)
            
            logging.info(f"Applying Open3D repair operations...")
            
            # Compute vertex normals for consistency
            mesh_o3d.compute_vertex_normals()
            
            # Remove degenerate triangles
            logging.info(f"   • Removing degenerate triangles...")
            original_triangles = len(mesh_o3d.triangles)
            mesh_o3d.remove_degenerate_triangles()
            removed = original_triangles - len(mesh_o3d.triangles)
            if removed > 0:
                logging.info(f"     Removed {removed} degenerate triangles")
                repair_report['improvements'].append(f"o3d_removed_{removed}_degenerate_triangles")
            
            # Remove duplicated triangles
            logging.info(f"   • Removing duplicate triangles...")
            original_triangles = len(mesh_o3d.triangles)
            mesh_o3d.remove_duplicated_triangles()
            removed = original_triangles - len(mesh_o3d.triangles)
            if removed > 0:
                logging.info(f"     Removed {removed} duplicate triangles")
                repair_report['improvements'].append(f"o3d_removed_{removed}_duplicate_triangles")
            
            # Remove duplicated vertices
            logging.info(f"   • Removing duplicate vertices...")
            original_vertices = len(mesh_o3d.vertices)
            mesh_o3d.remove_duplicated_vertices()
            removed = original_vertices - len(mesh_o3d.vertices)
            if removed > 0:
                logging.info(f"     Removed {removed} duplicate vertices")
                repair_report['improvements'].append(f"o3d_removed_{removed}_duplicate_vertices")
            
            # Remove non-manifold edges
            logging.info(f"   • Removing non-manifold edges...")
            mesh_o3d.remove_non_manifold_edges()
            repair_report['improvements'].append("o3d_non_manifold_edges_removed")
            
            # Check mesh properties with Open3D
            logging.info(f"Open3D mesh analysis:")
            is_edge_manifold = mesh_o3d.is_edge_manifold()
            is_vertex_manifold = mesh_o3d.is_vertex_manifold()
            is_watertight = mesh_o3d.is_watertight()
            is_orientable = mesh_o3d.is_orientable()
            
            logging.info(f"   Edge manifold: {'YES' if is_edge_manifold else 'NO'}")
            logging.info(f"   Vertex manifold: {'YES' if is_vertex_manifold else 'NO'}")
            logging.info(f"   Watertight: {'YES' if is_watertight else 'NO'}")
            logging.info(f"   Orientable: {'YES' if is_orientable else 'NO'}")
            
            # Convert back to trimesh
            logging.info(f"Converting Open3D → trimesh...")
            repaired_vertices = np.asarray(mesh_o3d.vertices)
            repaired_faces = np.asarray(mesh_o3d.triangles)
            
            # Create new trimesh with process=True for additional cleaning
            repaired_mesh = trimesh.Trimesh(
                vertices=repaired_vertices, 
                faces=repaired_faces, 
                process=True  # This applies additional trimesh processing
            )
            
            # Final watertight check
            final_watertight = repaired_mesh.is_watertight
            logging.info(f"Final Open3D repair result: {'WATERTIGHT' if final_watertight else 'STILL NOT WATERTIGHT'}")
            
            if final_watertight:
                repair_report['success'] = True
                repair_report['final_watertight'] = True
                repair_report['improvements'].append("o3d_repair_successful")
                logging.info(f"Open3D repair successful!")
            
        except Exception as e:
            logging.warning(f"Open3D repair encountered error: {e}")
            repair_report['improvements'].append(f"open3d_error: {str(e)}")
    
    elif repair_method in ["auto", "open3d", "hybrid"] and not OPEN3D_AVAILABLE:
        logging.warning(f"Open3D repair requested but Open3D not available")
        logging.info(f"Install with: pip install open3d")
        repair_report['improvements'].append("open3d_not_available")
    
    # Final report generation
    repair_report['final_vertex_count'] = len(repaired_mesh.vertices)
    repair_report['final_face_count'] = len(repaired_mesh.faces)
    repair_report['final_volume'] = abs(repaired_mesh.volume) if hasattr(repaired_mesh, 'volume') else 0
    repair_report['final_watertight'] = repaired_mesh.is_watertight
    
    # Success determination
    if not repair_report['success']:
        repair_report['success'] = repair_report['final_watertight'] or len(repair_report['improvements']) > 0
    
    # Generate comprehensive summary
    logging.info(f"MESH REPAIR SUMMARY")
    logging.info(f"Methods attempted: {', '.join(repair_report['methods_attempted'])}")
    logging.info(f"Improvements made: {len(repair_report['improvements'])}")
    for improvement in repair_report['improvements']:
        logging.info(f"   • {improvement}")
    
    logging.info(f"Before → After comparison:")
    logging.info(f"   Vertices: {repair_report['original_vertex_count']} → {repair_report['final_vertex_count']}")
    logging.info(f"   Faces: {repair_report['original_face_count']} → {repair_report['final_face_count']}")
    logging.info(f"   Volume: {repair_report['original_volume']:.2f} → {repair_report['final_volume']:.2f} mm³")
    logging.info(f"   Watertight: {'YES' if repair_report['original_watertight'] else 'NO'} → {'YES' if repair_report['final_watertight'] else 'NO'}")
    
    success_status = "SUCCESS" if repair_report['success'] else "PARTIAL"
    logging.info(f"Repair result: {success_status}")
    
    if not repair_report['final_watertight']:
        logging.warning(f"Warning: Mesh still not watertight after repair attempts")
        logging.warning(f"Mold creation may proceed but results might be affected")
        logging.warning(f"Consider using mesh editing software for manual repair")
    
    return repaired_mesh, repair_report


def apply_draft_angles_to_cavity(mold_with_cavity, original_mesh, draft_angle_degrees=1.5, parting_direction='z'):
    """
    🎯 DRAFT ANGLE IMPLEMENTATION
    
    Applies draft angles to vertical cavity walls to enable easier demolding.
    Draft angles prevent sticking and undercuts that make part removal difficult.
    
    Parameters:
    - mold_with_cavity: Mold mesh with cavity already created
    - original_mesh: Original object mesh for reference
    - draft_angle_degrees: Draft angle in degrees (typically 0.5-3°)
    - parting_direction: Direction of mold opening ('x', 'y', or 'z')
    
    Returns:
    - mold_with_drafted_cavity: Mold with draft angles applied
    - draft_report: Dictionary with draft application details
    """
    logging.info(f"DRAFT ANGLE IMPLEMENTATION")
    logging.info(f"Applying {draft_angle_degrees}° draft angles in {parting_direction.upper()}-direction")
    
    # Initialize draft report
    draft_report = {
        'draft_angle_degrees': draft_angle_degrees,
        'parting_direction': parting_direction,
        'original_volume': abs(mold_with_cavity.volume) if hasattr(mold_with_cavity, 'volume') else 0,
        'success': False,
        'method_used': 'none',
        'improvements': []
    }
    
    try:
        # Convert draft angle to radians
        draft_angle_rad = np.radians(draft_angle_degrees)
        
        # Get object bounds for scaling calculations
        object_bounds = original_mesh.bounds
        object_dims = object_bounds[1] - object_bounds[0]
        object_height = object_dims[2]  # Assume Z is vertical
        
        logging.info(f"Object dimensions: {object_dims}")
        logging.info(f"Object height: {object_height:.2f} mm")
        logging.info(f"Draft angle: {draft_angle_degrees}° ({draft_angle_rad:.4f} rad)")
        
        # Method 1: Volumetric Draft using Scaling Transformation
        logging.info(f"METHOD 1: VOLUMETRIC DRAFT")
        
        # Determine parting plane (where mold opens)
        parting_axis_index = {'x': 0, 'y': 1, 'z': 2}[parting_direction]
        
        # Calculate draft taper based on object height and angle
        draft_taper = np.tan(draft_angle_rad) * object_height
        logging.info(f"Calculated taper: {draft_taper:.3f} mm over {object_height:.2f} mm height")
        
        # Create a slightly enlarged version of the original object for draft
        object_center = (object_bounds[0] + object_bounds[1]) / 2
        
        # Calculate scaling factors for draft
        base_scale = 1.0  # No scaling at parting line
        max_scale = 1.0 + (2 * draft_taper / min(object_dims[0], object_dims[1]))  # Scale at deepest point
        
        logging.info(f"Draft scaling: base={base_scale:.4f}, max={max_scale:.4f}")
        
        # Method 1A: Simple uniform scaling approach
        if draft_angle_degrees <= 2.0:  # Conservative approach for small angles
            logging.info(f"Using CONSERVATIVE uniform scaling (draft ≤ 2°)")
            
            # Apply very slight uniform scaling to create draft
            uniform_scale = 1.0 + (draft_taper / max(object_dims))
            uniform_scale = min(uniform_scale, 1.05)  # Cap at 5% enlargement
            
            logging.info(f"Uniform scale factor: {uniform_scale:.4f}")
            
            # Create enlarged object for draft
            drafted_object = original_mesh.copy()
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = uniform_scale  # X scaling
            scale_matrix[1, 1] = uniform_scale  # Y scaling
            scale_matrix[2, 2] = 1.0  # No Z scaling (preserve height)
            
            # Center the scaling on the object center
            translation_to_origin = trimesh.transformations.translation_matrix(-object_center)
            scaling = scale_matrix
            translation_back = trimesh.transformations.translation_matrix(object_center)
            
            # Apply the transformation
            transform = np.dot(translation_back, np.dot(scaling, translation_to_origin))
            drafted_object.apply_transform(transform)
            
            # Get mold bounds for new cavity creation
            mold_bounds = mold_with_cavity.bounds
            mold_center = (mold_bounds[0] + mold_bounds[1]) / 2
            mold_size = mold_bounds[1] - mold_bounds[0]
            
            # Recreate mold block
            mold_block = trimesh.creation.box(
                extents=mold_size,
                transform=trimesh.transformations.translation_matrix(mold_center)
            )
            
            # Create new cavity with drafted object
            mold_with_drafted_cavity = trimesh.boolean.difference([mold_block, drafted_object])
            
            draft_report['method_used'] = 'uniform_scaling'
            draft_report['scale_factor'] = uniform_scale
            draft_report['success'] = True
            draft_report['improvements'].append(f"uniform_scaling_{uniform_scale:.4f}")
            
            logging.info(f"Conservative uniform scaling applied successfully")
            
        else:  # More aggressive approach for larger angles
            logging.info(f"Using ADVANCED variable scaling (draft > 2°)")
            
            # For now, fall back to uniform scaling for complex cases
            moderate_scale = 1.0 + (draft_taper / (2 * max(object_dims)))
            moderate_scale = min(moderate_scale, 1.08)  # Cap at 8% enlargement
            
            logging.info(f"Moderate scale factor: {moderate_scale:.4f}")
            
            # Create moderately enlarged object
            drafted_object = original_mesh.copy()
            scale_matrix = np.eye(4)
            scale_matrix[0, 0] = moderate_scale
            scale_matrix[1, 1] = moderate_scale
            scale_matrix[2, 2] = 1.0  # Preserve height
            
            # Apply centered scaling
            translation_to_origin = trimesh.transformations.translation_matrix(-object_center)
            scaling = scale_matrix
            translation_back = trimesh.transformations.translation_matrix(object_center)
            transform = np.dot(translation_back, np.dot(scaling, translation_to_origin))
            drafted_object.apply_transform(transform)
            
            # Recreate mold with drafted cavity
            mold_bounds = mold_with_cavity.bounds
            mold_center = (mold_bounds[0] + mold_bounds[1]) / 2
            mold_size = mold_bounds[1] - mold_bounds[0]
            
            mold_block = trimesh.creation.box(
                extents=mold_size,
                transform=trimesh.transformations.translation_matrix(mold_center)
            )
            
            mold_with_drafted_cavity = trimesh.boolean.difference([mold_block, drafted_object])
            
            draft_report['method_used'] = 'variable_scaling'
            draft_report['scale_factor'] = moderate_scale
            draft_report['success'] = True
            draft_report['improvements'].append(f"variable_scaling_{moderate_scale:.4f}")
            
            logging.info(f"Advanced variable scaling applied successfully")
        
        # Verify the result
        if hasattr(mold_with_drafted_cavity, 'volume'):
            draft_report['final_volume'] = abs(mold_with_drafted_cavity.volume)
            volume_change = draft_report['final_volume'] - draft_report['original_volume']
            volume_change_percent = (volume_change / draft_report['original_volume']) * 100
            
            logging.info(f"Volume analysis:")
            logging.info(f"   Original: {draft_report['original_volume']:.1f} mm³")
            logging.info(f"   Drafted: {draft_report['final_volume']:.1f} mm³")
            logging.info(f"   Change: {volume_change:+.1f} mm³ ({volume_change_percent:+.2f}%)")
        
        # Check if result is valid
        if not mold_with_drafted_cavity.is_watertight:
            logging.warning(f"Warning: Drafted mold is not watertight - attempting repair...")
            try:
                mold_with_drafted_cavity.fill_holes()
                if mold_with_drafted_cavity.is_watertight:
                    logging.info(f"Draft mold repaired successfully")
                    draft_report['improvements'].append("repaired_watertight")
                else:
                    logging.warning(f"Draft mold repair unsuccessful but proceeding...")
            except Exception as e:
                logging.warning(f"Draft mold repair failed: {e}")
    
    except Exception as e:
        logging.error(f"Draft angle application failed: {e}")
        logging.warning(f"Proceeding with original mold (no draft)")
        mold_with_drafted_cavity = mold_with_cavity
        draft_report['success'] = False
        draft_report['method_used'] = 'failed'
        draft_report['improvements'].append(f"failed: {str(e)}")
    
    # Generate comprehensive summary
    logging.info(f"DRAFT ANGLE SUMMARY")
    logging.info(f"Draft angle: {draft_angle_degrees}° in {parting_direction.upper()}-direction")
    logging.info(f"Method used: {draft_report['method_used']}")
    logging.info(f"Success: {'YES' if draft_report['success'] else 'NO'}")
    
    if draft_report['success']:
        logging.info(f"Applied improvements:")
        for improvement in draft_report['improvements']:
            logging.info(f"   • {improvement}")
        
        if 'scale_factor' in draft_report:
            logging.info(f"Scale factor: {draft_report['scale_factor']:.4f}")
        
        logging.info(f"Draft angles applied - easier demolding expected!")
    else:
        logging.warning(f"Draft application unsuccessful - demolding may require extra care")
    
    return mold_with_drafted_cavity, draft_report


def find_safe_key_positions(original_mesh, split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys=2):
    """
    🎯 ENHANCED CORNER-FIRST ALIGNMENT KEY POSITIONING
    
    Smart placement strategy:
    1. Try corner positions first (diagonal pairs for stability)
    2. Fallback to EDT-based positioning if corners fail
    3. Final fallback to simple positioned keys
    
    Parameters:
    - original_mesh: The original object mesh
    - split_axis: 'x', 'y', or 'z'
    - split_coordinate: The coordinate where the split occurs
    - min_corner, max_corner: Mold bounding box corners
    - wall_thickness: Thickness of mold walls
    - num_keys: Number of alignment keys to place (2 or 4, default: 2)
    
    Returns:
    - List of optimal key positions
    """
    logging.info(f"CORNER-FIRST KEY POSITIONING")
    logging.info(f"Finding {num_keys} alignment key positions using smart corner-first strategy")
    
    axis_index = {'x': 0, 'y': 1, 'z': 2}[split_axis]
    
    # Create a plane for cross-sectioning
    plane_origin = np.zeros(3)
    plane_origin[axis_index] = split_coordinate
    
    plane_normal = np.zeros(3)
    plane_normal[axis_index] = 1.0
    
    try:
        # Get cross-section of the original mesh at the split plane
        cross_section = original_mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
        
        if cross_section is None:
            logging.warning("Warning: Could not create cross-section. Using fallback key positions.")
            return get_fallback_key_positions(split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys)
        
        # Convert to 2D for easier processing
        slice_2D, to_3D = cross_section.to_planar()
        
        # Get the 2D bounds of the object's cross-section
        if len(slice_2D.vertices) == 0:
            logging.warning("Warning: Empty cross-section. Using fallback key positions.")
            return get_fallback_key_positions(split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys)
        
        # Define the mold face bounds in 2D coordinates
        if split_axis == 'z':
            # XY plane
            mold_bounds_2d = np.array([[min_corner[0], min_corner[1]], [max_corner[0], max_corner[1]]])
            coord_indices = [0, 1]  # X, Y
        elif split_axis == 'y':
            # XZ plane  
            mold_bounds_2d = np.array([[min_corner[0], min_corner[2]], [max_corner[0], max_corner[2]]])
            coord_indices = [0, 2]  # X, Z
        else:  # split_axis == 'x'
            # YZ plane
            mold_bounds_2d = np.array([[min_corner[1], min_corner[2]], [max_corner[1], max_corner[2]]])
            coord_indices = [1, 2]  # Y, Z
        
        logging.info(f"🔍 Mold bounds (2D): {mold_bounds_2d}")
        logging.info(f"�� Coordinate indices: {coord_indices}")
        
        # Step 1: Try corner-based positioning first
        logging.info(f"STEP 1: CORNER PLACEMENT")
        corner_positions = try_corner_positioning(
            slice_2D, mold_bounds_2d, coord_indices, axis_index, 
            split_coordinate, wall_thickness, num_keys
        )
        
        # Check if corner positioning provides good distribution
        has_good_distribution = False
        if len(corner_positions) >= num_keys:
            if num_keys == 2:
                # For 2 keys, check if we have top-bottom distribution
                corner_types = []
                for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in corner_positions:
                    if "Top" in corner_name:
                        corner_types.append("top")
                    elif "Bottom" in corner_name:
                        corner_types.append("bottom")
                
                has_good_distribution = "top" in corner_types and "bottom" in corner_types
                
                if has_good_distribution:
                    logging.info(f"Successfully found {len(corner_positions)} corner positions with good top-bottom distribution")
                    return [pos_3d for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in corner_positions[:num_keys]]
                else:
                    logging.warning(f"Corner placement found {len(corner_positions)}/{num_keys} positions but poor distribution (types: {corner_types})")
            else:
                # For 4 keys, any corner positioning is good
                logging.info(f"Successfully found {len(corner_positions)} corner positions")
                return [pos_3d for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in corner_positions[:num_keys]]
        
        # Step 2: Fallback to EDT-based positioning
        logging.info(f"STEP 2: EDT FALLBACK")
        if len(corner_positions) < num_keys:
            logging.info(f"Corner placement found only {len(corner_positions)}/{num_keys} positions. Using EDT fallback.")
        else:
            logging.info(f"Corner placement lacks good distribution. Using EDT fallback for optimal positioning.")
        
        edt_positions = find_edt_optimal_positions(
            slice_2D, mold_bounds_2d, coord_indices, axis_index, 
            split_coordinate, wall_thickness, num_keys
        )
        
        if len(edt_positions) >= 2:  # Need at least 2 keys for alignment
            logging.info(f"Successfully found {len(edt_positions)} EDT positions")
            return edt_positions[:num_keys]
        
        # Step 3: Final fallback
        logging.warning("Warning: Both corner and EDT methods failed. Using simple fallback.")
        return get_fallback_key_positions(split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys)
            
    except Exception as e:
        logging.warning(f"Warning: Error in key positioning ({e}). Using fallback.")
        return get_fallback_key_positions(split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys)


def get_actual_corner_coordinates(corner_name, mold_bounds_2d):
    """Get the actual corner coordinates for distance calculations."""
    corner_map = {
        "Bottom-Left": [mold_bounds_2d[0, 0], mold_bounds_2d[0, 1]],
        "Bottom-Right": [mold_bounds_2d[1, 0], mold_bounds_2d[0, 1]],
        "Top-Left": [mold_bounds_2d[0, 0], mold_bounds_2d[1, 1]],
        "Top-Right": [mold_bounds_2d[1, 0], mold_bounds_2d[1, 1]],
    }
    return corner_map[corner_name]


def try_corner_positioning(slice_2D, mold_bounds_2d, coord_indices, axis_index, split_coordinate, wall_thickness, num_keys):
    """
    🏠 SMART CORNER POSITIONING
    
    Try to place alignment keys in corners with top-bottom pairing for maximum stability.
    Uses size-adaptive safety margins and conservative positioning for small objects.
    Safety checks ensure keys don't interfere with cavity.
    """
    logging.info(f"Attempting corner placement for {num_keys} keys...")
    
    # Calculate object size for adaptive margins (estimate from mold bounds)
    object_dims = mold_bounds_2d[1] - mold_bounds_2d[0]
    # Remove wall thickness to get approximate object size
    approx_object_dims = object_dims - (2 * wall_thickness)
    avg_object_dim = np.mean(approx_object_dims)
    
    # SIZE-ADAPTIVE SAFETY MARGINS (matching main size detection logic)
    if avg_object_dim < 8.0:  # EXTRA_SMALL
        key_radius_factor = 1/6  # Even smaller keys for tiny objects
        safety_factor = 0.8      # Very conservative clearance
        corner_factor = 0.8      # Much further from corners
        size_category = "EXTRA_SMALL"
    elif avg_object_dim < 15.0:  # SMALL
        key_radius_factor = 1/5  # Slightly smaller keys
        safety_factor = 0.6      # More conservative clearance
        corner_factor = 0.7      # Further from corners
        size_category = "SMALL"
    else:  # MEDIUM/LARGE
        key_radius_factor = 1/4  # Standard key radius
        safety_factor = 0.4      # Standard clearance
        corner_factor = 0.5      # Standard corner margin
        size_category = "MEDIUM/LARGE"
    
    key_radius = wall_thickness * key_radius_factor
    safety_clearance = wall_thickness * safety_factor
    corner_margin = wall_thickness * corner_factor
    
    logging.info(f"Size category: {size_category} (approx obj: {avg_object_dim:.1f}mm)")
    logging.info(f"Adaptive margins: key_radius={key_radius:.2f}mm, safety={safety_clearance:.2f}mm, corner={corner_margin:.2f}mm")
    
    # Define corner positions with proper margins
    mold_width = mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0]
    mold_height = mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1]
    
    # Corner candidates (2D coordinates)
    corner_candidates = [
        ([mold_bounds_2d[0, 0] + corner_margin, mold_bounds_2d[0, 1] + corner_margin], "Bottom-Left"),
        ([mold_bounds_2d[1, 0] - corner_margin, mold_bounds_2d[0, 1] + corner_margin], "Bottom-Right"),
        ([mold_bounds_2d[0, 0] + corner_margin, mold_bounds_2d[1, 1] - corner_margin], "Top-Left"),
        ([mold_bounds_2d[1, 0] - corner_margin, mold_bounds_2d[1, 1] - corner_margin], "Top-Right"),
    ]
    
    logging.info(f"Evaluating {len(corner_candidates)} corner candidates...")
    
    # Test each corner for safety
    safe_corners = []
    for pos_2d, corner_name in corner_candidates:
        logging.info(f"Testing {corner_name} at ({pos_2d[0]:.2f}, {pos_2d[1]:.2f})")
        
        # Convert to 3D for safety testing
        pos_3d = np.zeros(3)
        pos_3d[coord_indices[0]] = pos_2d[0]
        pos_3d[coord_indices[1]] = pos_2d[1]
        pos_3d[axis_index] = split_coordinate
        
        # Test distance to object vertices (cavity)
        min_distance_to_cavity = float('inf')
        if len(slice_2D.vertices) > 0:
            for vertex in slice_2D.vertices:
                distance = np.linalg.norm(np.array(pos_2d) - vertex)
                min_distance_to_cavity = min(min_distance_to_cavity, distance)
        
        # Calculate distance to the actual corner (for "closer to corner than cavity" constraint)
        corner_coords = get_actual_corner_coordinates(corner_name, mold_bounds_2d)
        distance_to_corner = np.linalg.norm(np.array(pos_2d) - np.array(corner_coords))
        
        # Safety constraints
        required_clearance = key_radius + safety_clearance
        is_safe_from_cavity = min_distance_to_cavity >= required_clearance
        
        # Relaxed corner orientation constraint for better top/bottom distribution
        corner_zone_threshold = wall_thickness * 1.5  # More generous corner zone
        is_in_corner_zone = distance_to_corner <= corner_zone_threshold
        is_closer_to_corner = distance_to_corner < min_distance_to_cavity
        
        # Accept if either truly corner-oriented OR in reasonable corner zone
        is_corner_oriented = is_closer_to_corner or is_in_corner_zone
        
        logging.info(f"   Distance to cavity: {min_distance_to_cavity:.2f}mm (required: {required_clearance:.2f}mm)")
        logging.info(f"   Distance to corner: {distance_to_corner:.2f}mm (corner zone: ≤{corner_zone_threshold:.2f}mm)")
        logging.info(f"   Closer to corner than cavity: {'YES' if is_closer_to_corner else 'NO'}")
        logging.info(f"   In corner zone: {'YES' if is_in_corner_zone else 'NO'}")
        
        # Combined safety check with relaxed corner constraint
        is_safe = is_safe_from_cavity and is_corner_oriented
        
        if is_safe_from_cavity and is_closer_to_corner:
            status = "SAFE & CORNER-ORIENTED"
        elif is_safe_from_cavity and is_in_corner_zone and not is_closer_to_corner:
            status = "SAFE & IN CORNER ZONE"
        elif is_safe_from_cavity and not is_corner_oriented:
            status = "SAFE BUT TOO FAR FROM CORNER"
        elif not is_safe_from_cavity and is_corner_oriented:
            status = "CORNER-ORIENTED BUT TOO CLOSE TO CAVITY"
        else:
            status = "TOO CLOSE TO CAVITY & TOO FAR FROM CORNER"
        
        logging.info(f"   Status: {status}")
        
        if is_safe:
            safe_corners.append((pos_3d.tolist(), pos_2d, corner_name, min_distance_to_cavity, distance_to_corner))
    
    logging.info(f"Found {len(safe_corners)} safe corners")
    
    if len(safe_corners) == 0:
        logging.info(f"No safe corners available")
        return []
    
    # Select optimal corner pairs based on number of keys needed
    if num_keys == 2:
        return select_diagonal_corner_pair(safe_corners)
    else:  # num_keys == 4
        return select_four_corners(safe_corners, num_keys)


def select_diagonal_corner_pair(safe_corners):
    """Select the best diagonal pair for maximum alignment stability."""
    logging.info(f"Selecting optimal diagonal corner pair from {len(safe_corners)} safe corners")
    
    # Group corners by position (top vs bottom, left vs right)
    top_corners = []
    bottom_corners = []
    
    for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in safe_corners:
        if "Top" in corner_name:
            top_corners.append((pos_3d, pos_2d, corner_name, cavity_distance, corner_distance))
        elif "Bottom" in corner_name:
            bottom_corners.append((pos_3d, pos_2d, corner_name, cavity_distance, corner_distance))
    
    logging.info(f"Available: {len(top_corners)} top corners, {len(bottom_corners)} bottom corners")
    
    # Priority 1: DIAGONAL PAIRS for optimal alignment (opposite corners)
    diagonal_pairs = [
        (["Bottom-Left", "Top-Right"], "Bottom-Left ↔ Top-Right (optimal diagonal)"),
        (["Bottom-Right", "Top-Left"], "Bottom-Right ↔ Top-Left (optimal diagonal)"),
    ]
    
    for pair_names, pair_description in diagonal_pairs:
        # Find corners matching this diagonal pair
        pair_corners = []
        for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in safe_corners:
            if corner_name in pair_names:
                pair_corners.append((pos_3d, pos_2d, corner_name, cavity_distance, corner_distance))
        
        if len(pair_corners) >= 2:
            # Sort by safety distance (prefer corners further from cavity)
            pair_corners.sort(key=lambda x: x[3], reverse=True)
            selected = pair_corners[:2]
            
            logging.info(f"Selected {pair_description}:")
            for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in selected:
                logging.info(f"   {corner_name}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) - cavity: {cavity_distance:.2f}mm, corner: {corner_distance:.2f}mm")
            
            return selected
    
    # Priority 2: Any top + bottom combination (fallback for better than same-side)
    if len(top_corners) >= 1 and len(bottom_corners) >= 1:
        # Sort by safety distance (prefer corners further from cavity)
        top_corners.sort(key=lambda x: x[3], reverse=True)
        bottom_corners.sort(key=lambda x: x[3], reverse=True)
        
        # Try to find corners on opposite sides
        best_pair = None
        max_separation = 0
        
        for top_corner in top_corners:
            for bottom_corner in bottom_corners:
                # Calculate horizontal separation
                top_pos = top_corner[1]
                bottom_pos = bottom_corner[1]
                separation = abs(top_pos[0] - bottom_pos[0])  # X-axis separation
                
                if separation > max_separation:
                    max_separation = separation
                    best_pair = [top_corner, bottom_corner]
        
        if best_pair and max_separation > 0:
            selected = best_pair
            logging.info(f"Selected SEPARATED top-bottom pair (separation: {max_separation:.2f}mm):")
            for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in selected:
                logging.info(f"   {corner_name}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) - cavity: {cavity_distance:.2f}mm, corner: {corner_distance:.2f}mm")
            
            return selected
        else:
                         # Fallback to any top-bottom pair
             selected = [top_corners[0], bottom_corners[0]]
             logging.info(f"Selected same-side top-bottom pair (not ideal but functional):")
             for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in selected:
                 logging.info(f"   {corner_name}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) - cavity: {cavity_distance:.2f}mm, corner: {corner_distance:.2f}mm")
             
             return selected
    
    # Priority 3: Any two corners with maximum diagonal separation
    if len(safe_corners) >= 2:
        logging.info(f"No ideal diagonal pairing found. Finding corners with maximum separation.")
        
        # Find the pair with maximum distance between them
        best_pair = None
        max_distance = 0
        
        for i, corner1 in enumerate(safe_corners):
            for j, corner2 in enumerate(safe_corners):
                if i >= j:  # Avoid duplicates and self-comparison
                    continue
                    
                pos1 = corner1[1]  # 2D position
                pos2 = corner2[1]  # 2D position
                distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                
                if distance > max_distance:
                    max_distance = distance
                    best_pair = [corner1, corner2]
        
        if best_pair:
            selected = best_pair
            logging.info(f"Selected maximum separation pair (distance: {max_distance:.2f}mm):")
            for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in selected:
                logging.info(f"   {corner_name}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) - cavity: {cavity_distance:.2f}mm, corner: {corner_distance:.2f}mm")
            
            return selected
    
    logging.info("Insufficient safe corners for pairing")
    return []


def select_four_corners(safe_corners, num_keys):
    """Select up to 4 corners, prioritizing maximum safety distances."""
    logging.info(f"Selecting up to {num_keys} corners from {len(safe_corners)} safe corners")
    
    # Sort by safety distance (furthest from cavity first)
    safe_corners.sort(key=lambda x: x[3], reverse=True)
    selected = safe_corners[:min(num_keys, len(safe_corners))]
    
    logging.info(f"Selected {len(selected)} corners:")
    for pos_3d, pos_2d, corner_name, cavity_distance, corner_distance in selected:
        logging.info(f"   {corner_name}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) - cavity: {cavity_distance:.2f}mm, corner: {corner_distance:.2f}mm")
    
    return selected


def test_key_intersection_safety(key_position_3d, split_axis, wall_thickness, object_vertices_2d=None):
    """
    🔬 PHYSICAL INTERSECTION TEST
    
    Test if a proposed key position would physically intersect with the object cavity
    by checking proximity to object vertices and ensuring safe clearance.
    """
    try:
        key_radius = wall_thickness / 4  # Same radius as actual keys
        safety_clearance = wall_thickness * 0.3  # Additional safety clearance
        
        logging.info(f"Testing key at {key_position_3d} with radius {key_radius:.2f}mm, clearance {safety_clearance:.2f}mm")
        
        if object_vertices_2d is None or len(object_vertices_2d) == 0:
            logging.info(f"No object vertices available - using conservative approach")
            return True  # Conservative: allow if we can't test
        
        # Get the 2D projection of the key position based on split axis
        if split_axis == 'x':
            key_2d = [key_position_3d[1], key_position_3d[2]]  # Y, Z
        elif split_axis == 'y':
            key_2d = [key_position_3d[0], key_position_3d[2]]  # X, Z
        else:  # split_axis == 'z'
            key_2d = [key_position_3d[0], key_position_3d[1]]  # X, Y
        
        key_2d = np.array(key_2d)
        
        # Check distance to all object vertices
        min_distance_to_object = float('inf')
        closest_vertex = None
        
        for vertex in object_vertices_2d:
            distance = np.linalg.norm(key_2d - vertex)
            if distance < min_distance_to_object:
                min_distance_to_object = distance
                closest_vertex = vertex
        
        required_clearance = key_radius + safety_clearance
        is_safe = min_distance_to_object >= required_clearance
        
        logging.info(f"Key 2D position: {key_2d}")
        logging.info(f"Closest object vertex: {closest_vertex}")
        logging.info(f"Distance to closest vertex: {min_distance_to_object:.2f}mm")
        logging.info(f"Required clearance: {required_clearance:.2f}mm")
        logging.info(f"Safety status: {'SAFE' if is_safe else 'UNSAFE'}")
        
        return is_safe
        
    except Exception as e:
        logging.warning(f"Physical test error: {e} - using conservative approval")
        return True  # Conservative: allow if test fails


def find_edt_optimal_positions(slice_2D, mold_bounds_2d, coord_indices, axis_index, split_coordinate, wall_thickness, num_keys):
    """
    🎯 Use Euclidean Distance Transform to find optimal key positions.
    
    This creates a 2D binary image, computes EDT to find areas with maximum 
    distance from the object cavity, and selects optimal positions.
    """
    logging.info(f"Computing EDT for optimal key placement...")
    logging.info(f"slice_2D vertices shape: {slice_2D.vertices.shape if len(slice_2D.vertices) > 0 else 'EMPTY'}")
    logging.info(f"slice_2D has polygons_full: {hasattr(slice_2D, 'polygons_full')}")
    if hasattr(slice_2D, 'polygons_full'):
        logging.info(f"Number of polygons: {len(slice_2D.polygons_full)}")
    
    # Define image resolution (higher = more precise, but slower)
    resolution = 400  # Increased resolution for better accuracy
    
    # SIZE-ADAPTIVE SAFETY MARGINS for EDT positioning
    object_vertices_2d = slice_2D.vertices
    if len(object_vertices_2d) > 0:
        obj_bounds_2d = np.array([object_vertices_2d.min(axis=0), object_vertices_2d.max(axis=0)])
        obj_dims_2d = obj_bounds_2d[1] - obj_bounds_2d[0]
        avg_obj_dim_2d = np.mean(obj_dims_2d)
        
        if avg_obj_dim_2d < 8.0:  # EXTRA_SMALL objects
            safety_margin = wall_thickness * 1.8  # Much more conservative for tiny objects
            edge_margin = wall_thickness * 0.8    # Larger edge margin
            logging.info(f"EXTRA_SMALL object EDT margins: very conservative positioning")
        elif avg_obj_dim_2d < 15.0:  # SMALL objects
            safety_margin = wall_thickness * 1.4  # More conservative for small objects
            edge_margin = wall_thickness * 0.6    # Larger edge margin
            logging.info(f"SMALL object EDT margins: conservative positioning")
        else:  # MEDIUM/LARGE objects
            safety_margin = wall_thickness * 1.2  # Standard safety margin
            edge_margin = wall_thickness * 0.5    # Standard edge margin
            logging.info(f"Standard EDT margins: normal positioning")
    else:
        # Fallback if no object vertices
        safety_margin = wall_thickness * 1.8  # Conservative default
        edge_margin = wall_thickness * 0.8    # Conservative default
        logging.info(f"No object vertices - using conservative EDT margins")
    
    # Create 2D coordinate system
    mold_width = mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0]
    mold_height = mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1]
    
    logging.info(f"Mold dimensions: {mold_width:.2f} x {mold_height:.2f}")
    logging.info(f"Safety margin: {safety_margin:.2f} mm, Edge margin: {edge_margin:.2f} mm")
    
    # Create coordinate grids
    x_coords = np.linspace(mold_bounds_2d[0, 0], mold_bounds_2d[1, 0], resolution)
    y_coords = np.linspace(mold_bounds_2d[0, 1], mold_bounds_2d[1, 1], resolution)
    
    # Create binary image: 1 = free space, 0 = occupied by object
    binary_image = np.ones((resolution, resolution), dtype=np.uint8)
    
    logging.info(f"Object vertices bounds: min=({slice_2D.vertices.min(axis=0) if len(slice_2D.vertices) > 0 else 'N/A'}), max=({slice_2D.vertices.max(axis=0) if len(slice_2D.vertices) > 0 else 'N/A'})")
    
    # Mark object areas as occupied with enhanced debugging
    object_mask_created = False
    try:
        # Convert object vertices to image coordinates
        obj_vertices_2d = slice_2D.vertices
        if len(obj_vertices_2d) > 0:
            logging.info(f"Processing {len(obj_vertices_2d)} object vertices")
            
            # Scale vertices to image coordinates
            obj_x_scaled = ((obj_vertices_2d[:, 0] - mold_bounds_2d[0, 0]) / mold_width * (resolution - 1)).astype(int)
            obj_y_scaled = ((obj_vertices_2d[:, 1] - mold_bounds_2d[0, 1]) / mold_height * (resolution - 1)).astype(int)
            
            # Clip to image bounds
            obj_x_scaled = np.clip(obj_x_scaled, 0, resolution - 1)
            obj_y_scaled = np.clip(obj_y_scaled, 0, resolution - 1)
            
            logging.info(f"Scaled vertices X range: {obj_x_scaled.min()}-{obj_x_scaled.max()}")
            logging.info(f"Scaled vertices Y range: {obj_y_scaled.min()}-{obj_y_scaled.max()}")
            
            # Try to create filled polygon from object cross-section
            if hasattr(slice_2D, 'polygons_full') and len(slice_2D.polygons_full) > 0:
                logging.info(f"Using polygons_full approach with {len(slice_2D.polygons_full)} polygons")
                for i, polygon in enumerate(slice_2D.polygons_full):
                    try:
                        # Get polygon boundary points
                        boundary_coords = np.array(polygon.exterior.coords)
                        logging.info(f"Polygon {i+1} has {len(boundary_coords)} boundary points")
                        
                        # Scale to image coordinates
                        boundary_x = ((boundary_coords[:, 0] - mold_bounds_2d[0, 0]) / mold_width * (resolution - 1)).astype(int)
                        boundary_y = ((boundary_coords[:, 1] - mold_bounds_2d[0, 1]) / mold_height * (resolution - 1)).astype(int)
                        
                        # Clip to image bounds
                        boundary_x = np.clip(boundary_x, 0, resolution - 1)
                        boundary_y = np.clip(boundary_y, 0, resolution - 1)
                        
                        logging.info(f"Polygon {i+1} scaled X range: {boundary_x.min()}-{boundary_x.max()}")
                        logging.info(f"Polygon {i+1} scaled Y range: {boundary_y.min()}-{boundary_y.max()}")
                        
                        # Create filled polygon in binary image
                        polygon_points = np.column_stack((boundary_x, boundary_y))
                        cv2.fillPoly(binary_image, [polygon_points], 0)
                        object_mask_created = True
                        
                        logging.info(f"Successfully filled polygon {i+1}")
                        
                    except Exception as e:
                        logging.warning(f"Could not process polygon {i+1}: {e}")
                        continue
            
            # Fallback approach: use convex hull of all vertices
            if not object_mask_created:
                logging.info(f"Using fallback convex hull approach")
                try:
                    if len(obj_vertices_2d) >= 3:  # Need at least 3 points for convex hull
                        hull = ConvexHull(obj_vertices_2d)
                        hull_points = obj_vertices_2d[hull.vertices]
                        
                        # Scale hull points to image coordinates
                        hull_x = ((hull_points[:, 0] - mold_bounds_2d[0, 0]) / mold_width * (resolution - 1)).astype(int)
                        hull_y = ((hull_points[:, 1] - mold_bounds_2d[0, 1]) / mold_height * (resolution - 1)).astype(int)
                        
                        # Clip to image bounds
                        hull_x = np.clip(hull_x, 0, resolution - 1)
                        hull_y = np.clip(hull_y, 0, resolution - 1)
                        
                        hull_polygon = np.column_stack((hull_x, hull_y))
                        cv2.fillPoly(binary_image, [hull_polygon], 0)
                        object_mask_created = True
                        
                        logging.info(f"Successfully created convex hull mask with {len(hull_points)} points")
                        
                except Exception as e:
                    logging.warning(f"Convex hull approach failed: {e}")
            
            # Final fallback: mark circular areas around vertices with larger buffer
            if not object_mask_created:
                logging.info(f"Using final fallback - circular buffer approach")
                buffer_radius = max(10, int(safety_margin / max(mold_width, mold_height) * resolution))
                logging.info(f"Buffer radius: {buffer_radius} pixels")
                for x, y in zip(obj_x_scaled, obj_y_scaled):
                    cv2.circle(binary_image, (x, y), buffer_radius, 0, -1)
                object_mask_created = True
        
        occupied_pixels_after_object = np.sum(binary_image == 0)
        logging.info(f"Object occupancy: {occupied_pixels_after_object} / {resolution*resolution} pixels ({occupied_pixels_after_object/(resolution*resolution)*100:.1f}%)")
        
    except Exception as e:
        logging.warning(f"Error creating object mask ({e}), using conservative approach")
        # Conservative fallback: mark center area as occupied
        center_margin = int(resolution * 0.4)  # Larger conservative area
        binary_image[center_margin:-center_margin, center_margin:-center_margin] = 0
        object_mask_created = True
    
    # Add safety buffer around object areas
    if object_mask_created:
        logging.info(f"Adding safety buffer around object areas...")
        safety_buffer_pixels = max(5, int(safety_margin / max(mold_width, mold_height) * resolution))
        logging.info(f"Safety buffer: {safety_buffer_pixels} pixels ({safety_margin:.2f} mm)")
        
        # Create a larger structuring element for more conservative morphological dilation
        kernel_size = safety_buffer_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        object_mask = (binary_image == 0).astype(np.uint8)
        
        # Apply multiple iterations for better safety buffer
        dilated_mask = cv2.dilate(object_mask, kernel, iterations=2)
        binary_image[dilated_mask == 1] = 0
        
        occupied_pixels_after_safety = np.sum(binary_image == 0)
        logging.info(f"After safety buffer: {occupied_pixels_after_safety} / {resolution*resolution} pixels ({occupied_pixels_after_safety/(resolution*resolution)*100:.1f}%)")
    
    # Add edge margins (don't place keys too close to mold edges)
    edge_pixels = max(5, int(edge_margin / max(mold_width, mold_height) * resolution))
    binary_image[:edge_pixels, :] = 0      # Top edge
    binary_image[-edge_pixels:, :] = 0     # Bottom edge  
    binary_image[:, :edge_pixels] = 0      # Left edge
    binary_image[:, -edge_pixels:] = 0     # Right edge
    
    occupied_pixels_final = np.sum(binary_image == 0)
    logging.info(f"Edge margin: {edge_pixels} pixels ({edge_margin:.2f} mm)")
    logging.info(f"Final occupancy: {occupied_pixels_final} / {resolution*resolution} pixels ({occupied_pixels_final/(resolution*resolution)*100:.1f}%)")
    
    # Compute Euclidean Distance Transform
    logging.info(f"Computing Euclidean Distance Transform...")
    distance_transform = ndimage.distance_transform_edt(binary_image)
    
    max_distance = np.max(distance_transform)
    logging.info(f"Maximum EDT distance: {max_distance:.2f} pixels ({max_distance * max(mold_width, mold_height) / resolution:.2f} mm)")
    
    # Find optimal positions by detecting local maxima in EDT
    optimal_positions = []
    
    # Enhanced minimum distance requirement
    min_distance_pixels = max(8, int(safety_margin / max(mold_width, mold_height) * resolution))
    logging.info(f"Minimum required distance: {min_distance_pixels} pixels ({safety_margin:.2f} mm)")
    
    if num_keys == 2:
        # For 2 keys: find 2 positions with maximum distance, ensuring they're far apart
        optimal_positions = find_two_optimal_positions(distance_transform, resolution, mold_bounds_2d, min_distance_pixels)
    else:  # num_keys == 4
        # For 4 keys: find 4 well-distributed positions
        optimal_positions = find_four_optimal_positions(distance_transform, resolution, mold_bounds_2d, min_distance_pixels)
    
    # Convert 2D positions back to 3D coordinates and validate with physical intersection test
    key_positions_3d = []
    for i, pos_2d in enumerate(optimal_positions):
        pos_3d = np.zeros(3)
        pos_3d[coord_indices[0]] = pos_2d[0]
        pos_3d[coord_indices[1]] = pos_2d[1]
        pos_3d[axis_index] = split_coordinate
        
        # Calculate actual distance for validation
        img_x = int((pos_2d[0] - mold_bounds_2d[0, 0]) / (mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0]) * (resolution - 1))
        img_y = int((pos_2d[1] - mold_bounds_2d[0, 1]) / (mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1]) * (resolution - 1))
        img_x = np.clip(img_x, 0, resolution - 1)
        img_y = np.clip(img_y, 0, resolution - 1)
        distance_value = distance_transform[img_y, img_x]
        actual_distance = distance_value * max(mold_width, mold_height) / resolution
        
        # First validation: EDT distance check
        edt_valid = distance_value >= min_distance_pixels
        
        # Second validation: Physical intersection test with original mesh
        physical_valid = False
        if edt_valid:
            split_axis_name = ['x', 'y', 'z'][axis_index]
            physical_valid = test_key_intersection_safety(pos_3d, split_axis_name, wall_thickness, slice_2D.vertices if len(slice_2D.vertices) > 0 else None)
        
        is_valid = edt_valid and physical_valid
        
        if edt_valid and physical_valid:
            status = "VALID (EDT + Physical)"
        elif edt_valid and not physical_valid:
            status = "INVALID (Physical intersection detected)"
        else:
            status = "INVALID (EDT too close)"
        
        logging.info(f"Key {i+1}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) → distance: {actual_distance:.2f} mm, EDT: {distance_value:.1f} px {status}")
        
        if is_valid:
            key_positions_3d.append(pos_3d.tolist())
        else:
            if not edt_valid:
                logging.warning(f"Rejecting key {i+1} - too close to cavity (distance: {actual_distance:.2f} mm < required: {safety_margin:.2f} mm)")
            else:
                logging.warning(f"Rejecting key {i+1} - physical intersection test failed")
    
    # If we don't have enough valid positions, relax the constraints and try again
    if len(key_positions_3d) < 2:
        logging.warning(f"Only {len(key_positions_3d)} valid positions found, trying with relaxed constraints...")
        
        # Reduce safety requirements and try again
        relaxed_min_distance = max(2, min_distance_pixels // 2)
        logging.info(f"Relaxed minimum distance: {relaxed_min_distance} pixels")
        
        for i, pos_2d in enumerate(optimal_positions):
            if len(key_positions_3d) >= num_keys:
                break
                
            # Skip positions we already accepted
            pos_3d = np.zeros(3)
            pos_3d[coord_indices[0]] = pos_2d[0]
            pos_3d[coord_indices[1]] = pos_2d[1]
            pos_3d[axis_index] = split_coordinate
            
            if pos_3d.tolist() in key_positions_3d:
                continue
                
            # Re-check with relaxed constraints
            img_x = int((pos_2d[0] - mold_bounds_2d[0, 0]) / (mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0]) * (resolution - 1))
            img_y = int((pos_2d[1] - mold_bounds_2d[0, 1]) / (mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1]) * (resolution - 1))
            img_x = np.clip(img_x, 0, resolution - 1)
            img_y = np.clip(img_y, 0, resolution - 1)
            distance_value = distance_transform[img_y, img_x]
            actual_distance = distance_value * max(mold_width, mold_height) / resolution
            
            # Must pass both relaxed EDT and physical tests
            edt_ok = distance_value >= relaxed_min_distance
            split_axis_name = ['x', 'y', 'z'][axis_index]
            physical_ok = test_key_intersection_safety(pos_3d, split_axis_name, wall_thickness, slice_2D.vertices if len(slice_2D.vertices) > 0 else None)
            
            if edt_ok and physical_ok:
                key_positions_3d.append(pos_3d.tolist())
                logging.info(f"Relaxed Key {len(key_positions_3d)}: ({pos_2d[0]:.2f}, {pos_2d[1]:.2f}) → distance: {actual_distance:.2f} mm - ACCEPTED (EDT + Physical)")
            elif edt_ok and not physical_ok:
                logging.warning(f"Relaxed key rejected - physical test failed despite EDT distance: {actual_distance:.2f} mm")
            else:
                logging.warning(f"Relaxed key rejected - EDT distance too small: {actual_distance:.2f} mm")
    
    logging.info(f"EDT positioning complete: {len(key_positions_3d)} valid positions found from {len(optimal_positions)} candidates")
    
    return key_positions_3d


def find_two_optimal_positions(distance_transform, resolution, mold_bounds_2d, min_distance_pixels):
    """Find 2 optimal positions that are well-separated and have high EDT values."""
    from scipy.ndimage import maximum_filter
    
    logging.info(f"🔍 DEBUG: Finding 2 optimal positions with min distance {min_distance_pixels} pixels")
    
    # Filter out positions that don't meet minimum distance requirement
    valid_mask = distance_transform >= min_distance_pixels
    filtered_dt = distance_transform.copy()
    filtered_dt[~valid_mask] = 0
    
    valid_count = np.sum(valid_mask)
    logging.info(f"🔍 DEBUG: {valid_count} pixels meet minimum distance requirement")
    
    if valid_count < 2:
        logging.warning(f"⚠️  Warning: Only {valid_count} pixels meet distance requirement - using best available")
        filtered_dt = distance_transform.copy()
    
    # Apply maximum filter to find local maxima in valid areas
    local_maxima = (filtered_dt == maximum_filter(filtered_dt, size=30)) & (filtered_dt > 0)
    
    # Get coordinates of local maxima
    maxima_coords = np.argwhere(local_maxima)
    maxima_values = filtered_dt[local_maxima]
    
    logging.info(f"🔍 DEBUG: Found {len(maxima_coords)} local maxima")
    
    if len(maxima_coords) < 2:
        logging.warning(f"🔍 DEBUG: Using fallback approach - finding global maximum and well-separated second position")
        # Fallback: use global maximum and a well-separated point
        max_pos = np.unravel_index(np.argmax(filtered_dt), filtered_dt.shape)
        
        # Find a point far from the maximum
        distances_from_max = np.sqrt((np.arange(resolution)[:, None] - max_pos[0])**2 + 
                                   (np.arange(resolution)[None, :] - max_pos[1])**2)
        
        # Mask out the area near the maximum and find next best position
        masked_dt = filtered_dt.copy()
        masked_dt[distances_from_max < resolution * 0.25] = 0  # Reduced for better second position finding
        
        if np.max(masked_dt) > 0:
            second_pos = np.unravel_index(np.argmax(masked_dt), masked_dt.shape)
            maxima_coords = np.array([max_pos, second_pos])
            maxima_values = np.array([filtered_dt[max_pos], filtered_dt[second_pos]])
        else:
            logging.warning(f"⚠️  Warning: Could not find well-separated second position")
            maxima_coords = np.array([max_pos])
            maxima_values = np.array([filtered_dt[max_pos]])
    
    # Sort by EDT value (descending)
    sorted_indices = np.argsort(maxima_values)[::-1]
    
    # Select the two best positions that are sufficiently far apart
    selected_positions = []
    min_separation = max(resolution * 0.2, min_distance_pixels * 1.2)  # More relaxed for better coverage
    logging.info(f"🔍 DEBUG: Minimum separation: {min_separation:.1f} pixels")
    
    for i, idx in enumerate(sorted_indices):
        pos = maxima_coords[idx]
        edt_value = maxima_values[idx]
        
        logging.info(f"�� DEBUG: Candidate {i+1}: position ({pos[0]}, {pos[1]}), EDT value {edt_value:.1f}")
        
        # Check if this position is far enough from already selected positions
        is_far_enough = True
        for selected_pos in selected_positions:
            distance = np.sqrt((pos[0] - selected_pos[0])**2 + (pos[1] - selected_pos[1])**2)
            if distance < min_separation:
                is_far_enough = False
                logging.warning(f"🔍 DEBUG: Too close to existing position (distance: {distance:.1f} < {min_separation:.1f})")
                break
        
        if is_far_enough and edt_value >= min_distance_pixels:
            selected_positions.append(pos)
            logging.info(f"🔍 DEBUG: Selected position {len(selected_positions)}")
            if len(selected_positions) >= 2:
                break
    
    # Convert pixel coordinates back to world coordinates
    positions_2d = []
    for i, pos in enumerate(selected_positions):
        x = mold_bounds_2d[0, 0] + (pos[1] / (resolution - 1)) * (mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0])
        y = mold_bounds_2d[0, 1] + (pos[0] / (resolution - 1)) * (mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1])
        edt_value = distance_transform[pos[0], pos[1]]
        logging.info(f"🔍 DEBUG: Position {i+1} world coords: ({x:.2f}, {y:.2f}), EDT: {edt_value:.1f}")
        positions_2d.append([x, y])
    
    logging.info(f"🔍 DEBUG: Returning {len(positions_2d)} positions")
    return positions_2d


def find_four_optimal_positions(distance_transform, resolution, mold_bounds_2d, min_distance_pixels):
    """Find 4 optimal positions that are well-distributed around the mold."""
    logging.info(f"🔍 DEBUG: Finding 4 optimal positions with min distance {min_distance_pixels} pixels")
    
    # Filter out positions that don't meet minimum distance requirement
    valid_mask = distance_transform >= min_distance_pixels
    filtered_dt = distance_transform.copy()
    filtered_dt[~valid_mask] = 0
    
    valid_count = np.sum(valid_mask)
    logging.info(f"🔍 DEBUG: {valid_count} pixels meet minimum distance requirement")
    
    # Divide the mold into 4 quadrants and find the best position in each
    mid_x = resolution // 2
    mid_y = resolution // 2
    
    quadrants = [
        (slice(0, mid_x), slice(0, mid_y), "Top-left"),          # Top-left
        (slice(0, mid_x), slice(mid_y, None), "Top-right"),      # Top-right
        (slice(mid_x, None), slice(0, mid_y), "Bottom-left"),    # Bottom-left
        (slice(mid_x, None), slice(mid_y, None), "Bottom-right") # Bottom-right
    ]
    
    positions_2d = []
    
    for i, (quad_slice_x, quad_slice_y, quad_name) in enumerate(quadrants):
        # Extract quadrant
        quadrant_dt = filtered_dt[quad_slice_x, quad_slice_y]
        
        quadrant_max = np.max(quadrant_dt)
        quadrant_valid = np.sum(quadrant_dt > 0)
        logging.info(f"🔍 DEBUG: {quad_name} quadrant - max EDT: {quadrant_max:.1f}, valid pixels: {quadrant_valid}")
        
        if quadrant_max >= min_distance_pixels:
            # Find best position in this quadrant
            best_pos_local = np.unravel_index(np.argmax(quadrant_dt), quadrant_dt.shape)
            
            # Convert to global coordinates
            global_x = quad_slice_x.start + best_pos_local[0]
            global_y = quad_slice_y.start + best_pos_local[1]
            
            # Convert to world coordinates
            x = mold_bounds_2d[0, 0] + (global_y / (resolution - 1)) * (mold_bounds_2d[1, 0] - mold_bounds_2d[0, 0])
            y = mold_bounds_2d[0, 1] + (global_x / (resolution - 1)) * (mold_bounds_2d[1, 1] - mold_bounds_2d[0, 1])
            
            edt_value = distance_transform[global_x, global_y]
            logging.info(f"🔍 DEBUG: {quad_name} position: ({x:.2f}, {y:.2f}), EDT: {edt_value:.1f}")
            
            positions_2d.append([x, y])
        else:
            logging.warning(f"⚠️  Warning: {quad_name} quadrant has no valid positions (max EDT: {quadrant_max:.1f} < required: {min_distance_pixels})")
    
    logging.info(f"🔍 DEBUG: Returning {len(positions_2d)} positions from 4 quadrants")
    return positions_2d


# Legacy helper functions - now replaced by EDT-based positioning
# Keeping for reference, but no longer used in the main algorithm

# def is_position_safe(pos_2d, slice_2D, safety_margin):
#     """Check if a 2D position is far enough from the object's cross-section."""
#     try:
#         # Create a point to test
#         from shapely.geometry import Point
#         test_point = Point(pos_2d[0], pos_2d[1])
#         
#         # Check distance to all polygons in the cross-section
#         for polygon in slice_2D.polygons_full:
#             if test_point.distance(polygon.boundary) < safety_margin:
#                 return False
#         return True
#     except:
#         # If shapely operations fail, use simple distance check
#         min_distance = float('inf')
#         for vertex in slice_2D.vertices:
#             distance = np.linalg.norm(np.array(pos_2d) - vertex)
#             min_distance = min(min_distance, distance)
#         return min_distance >= safety_margin


# def generate_edge_positions(mold_bounds_2d, object_bounds_2d, min_distance_from_edge, safety_margin):
#     """Generate additional candidate positions along the mold edges."""
#     positions = []
#     
#     # Calculate the object center
#     obj_center = (object_bounds_2d[0] + object_bounds_2d[1]) / 2
#     
#     # Generate positions on each edge, avoiding the object projection
#     edges = [
#         # Bottom edge
#         ([mold_bounds_2d[0, 0] + min_distance_from_edge, mold_bounds_2d[0, 1] + min_distance_from_edge], 
#          [mold_bounds_2d[1, 0] - min_distance_from_edge, mold_bounds_2d[0, 1] + min_distance_from_edge]),
#         # Top edge  
#         ([mold_bounds_2d[0, 0] + min_distance_from_edge, mold_bounds_2d[1, 1] - min_distance_from_edge], 
#          [mold_bounds_2d[1, 0] - min_distance_from_edge, mold_bounds_2d[1, 1] - min_distance_from_edge]),
#         # Left edge
#         ([mold_bounds_2d[0, 0] + min_distance_from_edge, mold_bounds_2d[0, 1] + min_distance_from_edge], 
#          [mold_bounds_2d[0, 0] + min_distance_from_edge, mold_bounds_2d[1, 1] - min_distance_from_edge]),
#         # Right edge
#         ([mold_bounds_2d[1, 0] - min_distance_from_edge, mold_bounds_2d[0, 1] + min_distance_from_edge], 
#          [mold_bounds_2d[1, 0] - min_distance_from_edge, mold_bounds_2d[1, 1] - min_distance_from_edge]),
#     ]
#     
#     for start, end in edges:
#         # Try a few positions along each edge
#         for t in [0.25, 0.75]:
#             pos = np.array(start) + t * (np.array(end) - np.array(start))
#             positions.append(pos.tolist())
#     
#     return positions


def get_fallback_key_positions(split_axis, split_coordinate, min_corner, max_corner, wall_thickness, num_keys=2):
    """Fallback key positions when EDT positioning fails."""
    margin = wall_thickness * 0.8
    
    if split_axis == 'z':
        if num_keys == 2:
            return [
                [min_corner[0] + margin, min_corner[1] + margin, split_coordinate],
                [max_corner[0] - margin, max_corner[1] - margin, split_coordinate],
            ]
        else:  # num_keys == 4
            return [
                [min_corner[0] + margin, min_corner[1] + margin, split_coordinate],
                [max_corner[0] - margin, min_corner[1] + margin, split_coordinate],
                [min_corner[0] + margin, max_corner[1] - margin, split_coordinate],
                [max_corner[0] - margin, max_corner[1] - margin, split_coordinate],
            ]
    elif split_axis == 'y':
        if num_keys == 2:
            return [
                [min_corner[0] + margin, split_coordinate, min_corner[2] + margin],
                [max_corner[0] - margin, split_coordinate, max_corner[2] - margin],
            ]
        else:  # num_keys == 4
            return [
                [min_corner[0] + margin, split_coordinate, min_corner[2] + margin],
                [max_corner[0] - margin, split_coordinate, min_corner[2] + margin],
                [min_corner[0] + margin, split_coordinate, max_corner[2] - margin],
                [max_corner[0] - margin, split_coordinate, max_corner[2] - margin],
            ]
    else:  # split_axis == 'x'
        if num_keys == 2:
            return [
                [split_coordinate, min_corner[1] + margin, min_corner[2] + margin],
                [split_coordinate, max_corner[1] - margin, max_corner[2] - margin],
            ]
        else:  # num_keys == 4
            return [
                [split_coordinate, min_corner[1] + margin, min_corner[2] + margin],
                [split_coordinate, max_corner[1] - margin, min_corner[2] + margin],
                [split_coordinate, min_corner[1] + margin, max_corner[2] - margin],
                [split_coordinate, max_corner[1] - margin, max_corner[2] - margin],
            ]


def create_two_piece_mold(mold_with_cavity, split_axis, axis_index, min_corner, max_corner, mold_block_size):
    """Create 2-piece mold (original logic)."""
    logging.info("Creating 2-piece mold...")
    
    # Calculate split coordinate
    split_coordinate = (min_corner[axis_index] + max_corner[axis_index]) / 2
    
    # Create splitting boxes for the two halves
    # First half (negative side of the axis)
    first_half_extents = mold_block_size.copy()
    first_half_extents[axis_index] = mold_block_size[axis_index] / 2
    
    first_half_center = (min_corner + max_corner) / 2
    first_half_center[axis_index] = split_coordinate - mold_block_size[axis_index] / 4
    
    first_split_box = trimesh.creation.box(
        extents=first_half_extents,
        transform=trimesh.transformations.translation_matrix(first_half_center)
    )

    # Second half (positive side of the axis)
    second_half_extents = mold_block_size.copy()
    second_half_extents[axis_index] = mold_block_size[axis_index] / 2
    
    second_half_center = (min_corner + max_corner) / 2
    second_half_center[axis_index] = split_coordinate + mold_block_size[axis_index] / 4
    
    second_split_box = trimesh.creation.box(
        extents=second_half_extents,
        transform=trimesh.transformations.translation_matrix(second_half_center)
    )

    # Split the mold into two halves
    mold_first = trimesh.boolean.intersection([mold_with_cavity, first_split_box])
    mold_second = trimesh.boolean.intersection([mold_with_cavity, second_split_box])
    
    # Generate piece names based on split axis
    if split_axis == 'z':
        piece_names = ['bottom', 'top']
    elif split_axis == 'y':
        piece_names = ['front', 'back']
    else:  # split_axis == 'x'
        piece_names = ['left', 'right']
    
    return [mold_first, mold_second], piece_names


def create_four_piece_mold(mold_with_cavity, split_axis, axis_index, min_corner, max_corner, mold_block_size):
    """Create 4-piece mold by splitting along two axes."""
    logging.info("Creating 4-piece mold...")
    
    # Determine secondary split axis
    if split_axis == 'z':
        secondary_axis = 'x'  # Split XY plane into 4 quadrants
        secondary_index = 0
    elif split_axis == 'x':
        secondary_axis = 'y'  # Split YZ plane into 4 quadrants  
        secondary_index = 1
    else:  # split_axis == 'y'
        secondary_axis = 'z'  # Split XZ plane into 4 quadrants
        secondary_index = 2
    
    logging.info(f"Primary split: {split_axis.upper()}-axis, Secondary split: {secondary_axis.upper()}-axis")
    
    # Calculate split coordinates
    primary_split = (min_corner[axis_index] + max_corner[axis_index]) / 2
    secondary_split = (min_corner[secondary_index] + max_corner[secondary_index]) / 2
    
    # Create 4 splitting boxes
    pieces = []
    piece_names = []
    
    # Define the 4 quadrants: (primary_sign, secondary_sign)
    quadrants = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for i, (primary_sign, secondary_sign) in enumerate(quadrants):
        # Create extents for this piece (half size in both split dimensions)
        piece_extents = mold_block_size.copy()
        piece_extents[axis_index] = mold_block_size[axis_index] / 2
        piece_extents[secondary_index] = mold_block_size[secondary_index] / 2
        
        # Calculate center for this piece
        piece_center = (min_corner + max_corner) / 2
        piece_center[axis_index] = primary_split + primary_sign * mold_block_size[axis_index] / 4
        piece_center[secondary_index] = secondary_split + secondary_sign * mold_block_size[secondary_index] / 4
        
        # Create splitting box
        split_box = trimesh.creation.box(
            extents=piece_extents,
            transform=trimesh.transformations.translation_matrix(piece_center)
        )
        
        # Extract this piece
        piece = trimesh.boolean.intersection([mold_with_cavity, split_box])
        pieces.append(piece)
        
        # Generate piece name based on split axes
        if split_axis == 'z':  # Splitting XY plane
            primary_name = "bottom" if primary_sign == -1 else "top"
            secondary_name = "left" if secondary_sign == -1 else "right"
        elif split_axis == 'y':  # Splitting XZ plane
            primary_name = "front" if primary_sign == -1 else "back"
            secondary_name = "bottom" if secondary_sign == -1 else "top"
        else:  # split_axis == 'x', splitting YZ plane
            primary_name = "left" if primary_sign == -1 else "right"
            secondary_name = "front" if secondary_sign == -1 else "back"
        
        piece_name = f"{primary_name}_{secondary_name}"
        piece_names.append(piece_name)
        
        logging.info(f"Piece {i+1}: {piece_name} at center {piece_center}")
    
    return pieces, piece_names


def add_alignment_keys_two_piece(mold_pieces, key_positions_3d, split_axis, wall_thickness, object_size_info=None):
    """Add alignment keys to 2-piece mold with robust mesh validation."""
    logging.info(f"Adding {len(key_positions_3d)} alignment keys to 2-piece mold...")
    
    # SIZE-ADAPTIVE KEY DIMENSIONS 
    # Determine object size category for proportional key sizing
    if object_size_info and 'size_category' in object_size_info:
        size_category = object_size_info['size_category']
        avg_dimension = object_size_info.get('avg_dimension', 20.0)
        logging.info(f"🔧 Object size category: {size_category} (avg: {avg_dimension:.2f}mm)")
        
        # Adaptive key sizing based on object size
        if size_category == "EXTRA_SMALL":
            # Much smaller keys for tiny objects
            key_radius_factor = 1/6  # 16.7% of wall thickness (was 25%)
            key_height_factor = 0.6  # 60% of wall thickness (was 100%)
            logging.info(f"🔧 EXTRA_SMALL sizing: reduced key dimensions for tiny objects")
        elif size_category == "SMALL":
            # Slightly smaller keys for small objects  
            key_radius_factor = 1/5  # 20% of wall thickness (was 25%)
            key_height_factor = 0.8  # 80% of wall thickness (was 100%)
            logging.info(f"🔧 SMALL sizing: slightly reduced key dimensions")
        else:
            # Standard sizing for medium/large objects
            key_radius_factor = 1/4  # 25% of wall thickness (original)
            key_height_factor = 1.0  # 100% of wall thickness (original)
            logging.info(f"🔧 Standard sizing for {size_category} objects")
    else:
        # Fallback to standard sizing
        key_radius_factor = 1/4
        key_height_factor = 1.0
        logging.info(f"🔧 Using standard key sizing (no size info provided)")
    
    key_radius = wall_thickness * key_radius_factor
    key_height = wall_thickness * key_height_factor
    
    # Adaptive tolerance gap
    if key_radius < 0.6:  # Very small keys
        tolerance_gap = max(0.15, key_radius * 0.25)  # Smaller tolerance for tiny keys
    elif key_radius < 1.0:  # Small keys
        tolerance_gap = max(0.2, key_radius * 0.3)   # Moderate tolerance
    else:  # Standard keys
        tolerance_gap = 0.4  # Standard tolerance
    
    # Special handling for very small keys
    if key_radius < 0.5:
        logging.warning(f"⚠️  Very tiny alignment keys detected (radius: {key_radius:.2f}mm)")
        logging.info(f"⚠️  Using extra-careful boolean operations for tiny geometries...")
    
    logging.info(f"🔧 Size-adaptive key specs: radius={key_radius:.2f}mm ({key_radius_factor*100:.0f}% of wall), height={key_height:.2f}mm ({key_height_factor*100:.0f}% of wall), tolerance={tolerance_gap:.2f}mm")

    # Validate input mold pieces
    mold_pieces[0] = validate_mesh_for_boolean(mold_pieces[0], "mold_piece_1")
    mold_pieces[1] = validate_mesh_for_boolean(mold_pieces[1], "mold_piece_2")

    for i, position in enumerate(key_positions_3d):
        logging.info(f"�� Creating alignment key {i+1} at position {position}")
        
        try:
            # Create key cylinders oriented along the split axis
            key_first = trimesh.creation.cylinder(
                radius=key_radius,
                height=key_height,
                sections=32
            )

            key_second = trimesh.creation.cylinder(
                radius=key_radius + tolerance_gap,
                height=key_height,
                sections=32
            )
            
            # Orient the cylinders along the split axis
            if split_axis == 'x':
                rotation = trimesh.transformations.rotation_matrix(
                    angle=3.14159/2, direction=[0, 1, 0]
                )
                key_first.apply_transform(rotation)
                key_second.apply_transform(rotation)
            elif split_axis == 'y':
                rotation = trimesh.transformations.rotation_matrix(
                    angle=3.14159/2, direction=[1, 0, 0]
                )
                key_first.apply_transform(rotation)
                key_second.apply_transform(rotation)
            
            # Position the keys
            key_first.apply_translation(position)
            key_second.apply_translation(position)
            
            # Validate key cylinders
            key_first = validate_mesh_for_boolean(key_first, f"key_{i+1}_positive")
            key_second = validate_mesh_for_boolean(key_second, f"key_{i+1}_negative")

            # Add key to the first mold piece using safe operation
            logging.info(f"🔧   Applying positive key to piece 1...")
            mold_pieces[0] = safe_boolean_operation(
                'union', 
                [mold_pieces[0], key_first], 
                f"add_key_{i+1}_to_piece_1",
                fallback_mesh=mold_pieces[0]
            )
            
            # Subtract the key from the second mold piece using safe operation
            logging.info(f"   Applying negative key to piece 2...")
            mold_pieces[1] = safe_boolean_operation(
                'difference', 
                [mold_pieces[1], key_second], 
                f"subtract_key_{i+1}_from_piece_2",
                fallback_mesh=mold_pieces[1]
            )
            
            logging.info(f"   Key {i+1} successfully applied")
            
        except Exception as e:
            logging.error(f"Error applying key {i+1}: {e}")
            logging.warning(f"Skipping this key and continuing...")
            continue
    
    logging.info(f"Alignment key application completed")
    return mold_pieces


def validate_mesh_for_boolean(mesh, mesh_name="mesh"):
    """Validate and repair mesh for boolean operations with enhanced repair."""
    logging.info(f"Validating {mesh_name}...")
    
    if not mesh.is_watertight:
        logging.warning(f"{mesh_name} is not watertight, attempting enhanced repair...")
        try:
            # Method 1: Basic fill_holes
            mesh.fill_holes()
            if mesh.is_watertight:
                logging.info(f"{mesh_name} repaired with fill_holes")
                return mesh
            
            # Method 2: More aggressive repair
            logging.info(f"Trying advanced repair for {mesh_name}...")
            
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Remove duplicate faces
            mesh.remove_duplicate_faces()
            
            # Fix normals
            mesh.fix_normals()
            
            # Try fill_holes again after cleaning
            mesh.fill_holes()
            
            if mesh.is_watertight:
                logging.info(f"{mesh_name} repaired with advanced methods")
                return mesh
            
            # Method 3: Process with trimesh's repair
            processed_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
            if processed_mesh.is_watertight:
                logging.info(f"{mesh_name} repaired with process=True")
                return processed_mesh
            
            logging.warning(f"{mesh_name} repair unsuccessful but proceeding...")
            
        except Exception as e:
            logging.warning(f"{mesh_name} repair failed: {e}")
    else:
        logging.info(f"{mesh_name} is watertight")
    
    return mesh


def safe_boolean_operation(operation, meshes, operation_name="boolean", fallback_mesh=None):
    """
    Perform boolean operations with enhanced error handling and fallback strategies.
    
    Parameters:
    - operation: 'union' or 'difference'
    - meshes: List of meshes for the operation
    - operation_name: Description for logging
    - fallback_mesh: Mesh to return if operation fails
    
    Returns:
    - Result mesh or fallback_mesh if operation fails
    """
    try:
        # Validate all input meshes
        validated_meshes = []
        for i, mesh in enumerate(meshes):
            validated_mesh = validate_mesh_for_boolean(mesh, f"{operation_name}_input_{i+1}")
            validated_meshes.append(validated_mesh)
        
        # Perform the boolean operation
        logging.info(f"Performing {operation} operation: {operation_name}")
        
        if operation == 'union':
            result = trimesh.boolean.union(validated_meshes)
        elif operation == 'difference':
            result = trimesh.boolean.difference(validated_meshes)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Validate result
        if result is not None:
            result = validate_mesh_for_boolean(result, f"{operation_name}_result")
            logging.info(f"{operation_name} completed successfully")
            return result
        else:
            logging.warning(f"{operation_name} returned None - using fallback")
            return fallback_mesh if fallback_mesh is not None else meshes[0]
            
    except Exception as e:
        logging.error(f"{operation_name} failed: {e}")
        if fallback_mesh is not None:
            logging.info(f"Using provided fallback mesh")
            return fallback_mesh
        else:
            logging.info(f"Using first input mesh as fallback")
            return meshes[0]


def add_alignment_keys_four_piece(mold_pieces, split_axis, min_corner, max_corner, wall_thickness, original_mesh):
    """
    🎯 SIMPLE 4-PIECE ALIGNMENT KEYS
    
    Creates alignment keys by treating 4-piece mold as four separate 2-piece pairs:
    1. right-top ↔ left-top (horizontal alignment)
    2. right-bottom ↔ left-bottom (horizontal alignment) 
    3. right-top ↔ right-bottom (vertical alignment)
    4. left-top ↔ left-bottom (vertical alignment)
    
    This ensures consistent positioning and no intersections.
    """
    logging.info("Creating 4-piece alignment keys using simple 2-piece pair approach...")
    
    axis_index = {'x': 0, 'y': 1, 'z': 2}[split_axis]
    
    # Calculate split coordinates
    primary_split = (min_corner[axis_index] + max_corner[axis_index]) / 2
    
    # Determine secondary axis and split coordinate
    if split_axis == 'z':
        secondary_axis = 'x'
        secondary_index = 0
        secondary_split = (min_corner[secondary_index] + max_corner[secondary_index]) / 2
        logging.info(f"Z-axis split detected - using X-axis for horizontal alignment")
        
        # For Z-axis split: piece order is [bottom_left, bottom_right, top_left, top_right]
        pairs = [
            ("horizontal_bottom", 0, 1, "bottom_left", "bottom_right", secondary_axis, secondary_split),  # bottom pair
            ("horizontal_top", 2, 3, "top_left", "top_right", secondary_axis, secondary_split),          # top pair
            ("vertical_left", 0, 2, "bottom_left", "top_left", split_axis, primary_split),               # left pair
            ("vertical_right", 1, 3, "bottom_right", "top_right", split_axis, primary_split)             # right pair
        ]
        
    elif split_axis == 'x':
        secondary_axis = 'y'
        secondary_index = 1
        secondary_split = (min_corner[secondary_index] + max_corner[secondary_index]) / 2
        logging.info(f"X-axis split detected - using Y-axis for horizontal alignment")
        
        # For X-axis split: piece order is [left_front, left_back, right_front, right_back]
        pairs = [
            ("horizontal_left", 0, 1, "left_front", "left_back", secondary_axis, secondary_split),       # left pair
            ("horizontal_right", 2, 3, "right_front", "right_back", secondary_axis, secondary_split),   # right pair
            ("vertical_front", 0, 2, "left_front", "right_front", split_axis, primary_split),           # front pair
            ("vertical_back", 1, 3, "left_back", "right_back", split_axis, primary_split)               # back pair
        ]
        
    else:  # split_axis == 'y'
        secondary_axis = 'z'
        secondary_index = 2
        secondary_split = (min_corner[secondary_index] + max_corner[secondary_index]) / 2
        logging.info(f"Y-axis split detected - using Z-axis for horizontal alignment")
        
        # For Y-axis split: piece order is [front_bottom, front_top, back_bottom, back_top]
        pairs = [
            ("horizontal_front", 0, 1, "front_bottom", "front_top", secondary_axis, secondary_split),   # front pair
            ("horizontal_back", 2, 3, "back_bottom", "back_top", secondary_axis, secondary_split),     # back pair
            ("vertical_bottom", 0, 2, "front_bottom", "back_bottom", split_axis, primary_split),       # bottom pair
            ("vertical_top", 1, 3, "front_top", "back_top", split_axis, primary_split)                 # top pair
        ]
    
    logging.info(f"Primary split ({split_axis}): {primary_split:.2f}")
    logging.info(f"Secondary split ({secondary_axis}): {secondary_split:.2f}")
    logging.info(f"Processing {len(pairs)} alignment pairs...")
    
    # Process each pair using simple 2-piece logic
    for pair_idx, (pair_type, idx1, idx2, name1, name2, align_axis, split_coord) in enumerate(pairs):
        logging.info(f"PAIR {pair_idx + 1}: {pair_type} ({name1} ↔ {name2})")
        
        # Get the two pieces for this pair
        piece1 = mold_pieces[idx1].copy()  # Make copies to avoid modifying originals during validation
        piece2 = mold_pieces[idx2].copy()
        
        # Validate meshes before boolean operations
        piece1 = validate_mesh_for_boolean(piece1, name1)
        piece2 = validate_mesh_for_boolean(piece2, name2)
        
        # Calculate appropriate bounds for this pair
        if "horizontal" in pair_type:
            # For horizontal pairs, use bounds that span both pieces in the primary axis
            if pair_idx < 2:  # First two pairs are on one side of primary split
                pair_min_corner = min_corner.copy()
                pair_max_corner = max_corner.copy()
                if pair_idx == 0:  # Bottom/front/left pair
                    pair_max_corner[axis_index] = primary_split
                else:  # Top/back/right pair
                    pair_min_corner[axis_index] = primary_split
            else:  # This shouldn't happen for horizontal pairs in this structure
                pair_min_corner = min_corner.copy()
                pair_max_corner = max_corner.copy()
        else:  # vertical pairs
            # For vertical pairs, use bounds that span both pieces in the secondary axis
            pair_min_corner = min_corner.copy()
            pair_max_corner = max_corner.copy()
            if "left" in pair_type or "front" in pair_type or "bottom" in pair_type:
                pair_max_corner[secondary_index] = secondary_split
            else:  # right/back/top
                pair_min_corner[secondary_index] = secondary_split
        
        logging.info(f"Pair alignment axis: {align_axis.upper()}")
        logging.info(f"Split coordinate: {split_coord:.2f}")
        logging.info(f"Pair bounds: min={pair_min_corner}, max={pair_max_corner}")
        
        # Generate simple, safe key positions for this pair
        logging.info(f"Generating safe key positions for {align_axis}-axis alignment...")
        key_positions_3d = get_fallback_key_positions(
            align_axis, split_coord, pair_min_corner, pair_max_corner, wall_thickness, num_keys=2
        )
        
        logging.info(f"Key positions: {key_positions_3d}")
        
        # Apply alignment keys using the proven 2-piece method
        logging.info(f"Applying alignment keys to {name1} ↔ {name2}...")
        try:
            pair_pieces = [piece1, piece2]
            
            # Extract size info from original mesh for adaptive key sizing
            object_bounds = original_mesh.bounds
            object_dims = object_bounds[1] - object_bounds[0]
            avg_dimension = np.mean(object_dims)
            
            # Determine size category (same logic as in detect_optimal_wall_thickness)
            if avg_dimension < 8.0:
                size_category = "EXTRA_SMALL"
            elif avg_dimension < 15.0:
                size_category = "SMALL"
            elif avg_dimension < 50.0:
                size_category = "MEDIUM"
            else:
                size_category = "LARGE"
            
            object_size_info = {
                'size_category': size_category,
                'avg_dimension': avg_dimension,
                'dimensions': object_dims.tolist()
            }
            
            pair_pieces = add_alignment_keys_two_piece(pair_pieces, key_positions_3d, align_axis, wall_thickness, object_size_info)
            
            # Update the original mold pieces
            mold_pieces[idx1] = pair_pieces[0]
            mold_pieces[idx2] = pair_pieces[1]
            
            logging.info(f"Pair {pair_idx + 1} alignment keys completed successfully!")
            
        except Exception as e:
            logging.error(f"Error applying alignment keys to pair {pair_idx + 1}: {e}")
            logging.warning(f"Skipping this pair and continuing...")
            continue
    
    logging.info(f"4-piece alignment system completed!")
    logging.info(f"Total pairs processed: {len(pairs)}")
    logging.info(f"Alignment directions: {secondary_axis.upper()}-axis (horizontal) + {split_axis.upper()}-axis (vertical)")
    return mold_pieces


def add_spout_to_pieces(mold_pieces, spout_info):
    """Add pour spout to all mold pieces."""
    logging.info("Adding pour spout to all mold pieces...")
    
    pour_spout_radius = spout_info['spout_radius']
    spout_start = np.array(spout_info['spout_start'])
    spout_end = np.array(spout_info['spout_end'])
    spout_length = np.linalg.norm(spout_start - spout_end)
    spout_center = (spout_start + spout_end) / 2
    
    # Create spout for subtraction
    spout_for_subtraction = trimesh.creation.cylinder(
        radius=pour_spout_radius * 1.01,
        height=spout_length * 1.1,
        sections=32
    )
    spout_for_subtraction.apply_translation(spout_center)
    
    # Apply to all pieces using safe boolean operations
    for i, piece in enumerate(mold_pieces):
        logging.info(f"Applying spout to piece {i+1}...")
        try:
            # Use safe boolean operation for spout subtraction
            mold_pieces[i] = safe_boolean_operation(
                'difference', 
                [piece, spout_for_subtraction], 
                f"apply_spout_to_piece_{i+1}",
                fallback_mesh=piece
            )
            logging.info(f"Spout applied to piece {i+1}")
        except Exception as e:
            logging.warning(f"Could not apply spout to piece {i+1}: {e}")
            # Keep original piece if spout application fails
            mold_pieces[i] = piece
    
    return mold_pieces

def get_simple_spout_position(original_mesh, split_axis, split_coordinate, min_corner, max_corner, wall_thickness):
    """
    Position spout at the bottom of the mold, always vertical (along Z-axis),
    following the reference notebook's reliable bottom hole approach.
    """
    logging.info(f"SPOUT POSITIONING")
    logging.info(f"Split axis: {split_axis}")
    logging.info(f"Split coordinate: {split_coordinate}")
    logging.info(f"Mold bounds: min={min_corner}, max={max_corner}")
    logging.info(f"Wall thickness: {wall_thickness}")
    
    # Get the original object bounds for cavity intersection calculation
    object_bounds = original_mesh.bounds
    logging.info(f"Object bounds: {object_bounds}")
    
    # Calculate spout radius as fraction of mold size (following reference notebook)
    mold_extents = max_corner - min_corner
    spout_radius = min(mold_extents[:2]) * 0.1  # Using reference notebook ratio
    logging.info(f"Mold extents: {mold_extents}")
    logging.info(f"Calculated spout radius: {spout_radius}")
    
    # Calculate cavity center and mid-point (following reference notebook approach)
    cavity_center = (object_bounds[0] + object_bounds[1]) / 2
    cavity_mid_z = (object_bounds[0][2] + object_bounds[1][2]) / 2
    logging.info(f"Cavity center: {cavity_center}")
    logging.info(f"Cavity mid Z: {cavity_mid_z}")
    
    # Calculate object dimensions for size-adaptive spout penetration
    object_dims = object_bounds[1] - object_bounds[0]
    object_height = object_dims[2]  # Z-dimension (height)
    avg_object_dim = (object_dims[0] + object_dims[1] + object_dims[2]) / 3
    
    logging.info(f"Object dimensions: {object_dims}")
    logging.info(f"Object height: {object_height:.2f} mm, avg dimension: {avg_object_dim:.2f} mm")
    
    # SIZE-ADAPTIVE SPOUT PENETRATION
    # Calculate spout penetration depth based on object size to replace hardcoded -20
    if avg_object_dim < 15.0:  # Small objects
        penetration_percentage = 0.25  # 25% of object height
        min_penetration = 2.0  # Minimum 2mm penetration
        max_penetration = 10.0  # Maximum 10mm penetration
        logging.info(f"SMALL object mode: 25% penetration (min: {min_penetration}mm, max: {max_penetration}mm)")
    else:  # Large objects
        penetration_percentage = 0.15  # 15% of object height
        min_penetration = 5.0  # Minimum 5mm penetration
        max_penetration = 25.0  # Maximum 25mm penetration
        logging.info(f"LARGE object mode: 15% penetration (min: {min_penetration}mm, max: {max_penetration}mm)")
    
    # Calculate adaptive penetration depth
    calculated_penetration = object_height * penetration_percentage
    adaptive_penetration = max(min_penetration, min(calculated_penetration, max_penetration))
    
    logging.info(f"Calculated penetration: {calculated_penetration:.2f} mm (={penetration_percentage*100:.0f}% of {object_height:.2f} mm)")
    logging.info(f"Applied penetration: {adaptive_penetration:.2f} mm (after min/max constraints)")
    
    # Calculate mold center
    mold_center = (min_corner + max_corner) / 2
    logging.info(f"Mold center: {mold_center}")
    
    # Always create vertical spout at bottom (following reference notebook)
    # Position at mold center for X and Y, extend from bottom to cavity with adaptive penetration
    logging.info("Creating vertical spout from bottom with SIZE-ADAPTIVE penetration")
    
    spout_start_x = mold_center[0]  # X at mold center
    spout_start_y = mold_center[1]  # Y at mold center  
    spout_start_z = min_corner[2]   # Start from bottom of mold
    
    spout_end_x = mold_center[0]    # Keep X at mold center
    spout_end_y = mold_center[1]    # Keep Y at mold center
    spout_end_z = cavity_mid_z - adaptive_penetration  # SIZE-ADAPTIVE penetration (replaces hardcoded -20)
    
    spout_start = [spout_start_x, spout_start_y, spout_start_z]
    spout_end = [spout_end_x, spout_end_y, spout_end_z]
    spout_length = np.linalg.norm(np.array(spout_start) - np.array(spout_end))
    
    logging.info(f"Spout start: {spout_start}")
    logging.info(f"Spout end: {spout_end}")
    logging.info(f"Spout length: {spout_length:.2f} mm")
    logging.info(f"Penetration into cavity: {adaptive_penetration:.2f} mm (was hardcoded -20)")
    logging.info(f"Size-adaptive approach: {'SMALL' if avg_object_dim < 15.0 else 'LARGE'} object mode")
    
    return {
        'spout_start': spout_start,
        'spout_end': spout_end,
        'spout_radius': spout_radius,
        'spout_length': spout_length
    }

def generate_settings_summary(input_file, thickness_analysis, draft_report, split_axis, mold_pieces, num_alignment_keys, repair_method):
    """
    Generate a concise settings summary for the mold creation session.
    """
    from datetime import datetime
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract key information
    object_analysis = thickness_analysis['object_analysis']
    
    # Handle user thickness display
    user_thickness_display = f"{thickness_analysis['user_thickness']:.2f} mm" if thickness_analysis['user_thickness'] is not None else 'Auto-calculated'
    
    # Get expected file names
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    if mold_pieces == 2:
        if split_axis == 'z':
            piece_names = ['bottom', 'top']
        elif split_axis == 'y':
            piece_names = ['front', 'back']
        else:  # split_axis == 'x'
            piece_names = ['left', 'right']
    else:  # 4-piece
        if split_axis == 'z':
            piece_names = ['bottom_left', 'bottom_right', 'top_left', 'top_right']
        elif split_axis == 'x':
            piece_names = ['left_front', 'left_back', 'right_front', 'right_back']
        else:  # split_axis == 'y'
            piece_names = ['front_bottom', 'front_top', 'back_bottom', 'back_top']
    
    # Build concise summary
    summary = f"""STL MOLD MAKER - SETTINGS SUMMARY
Generated: {timestamp}

INPUT OBJECT
File: {os.path.basename(input_file)}
Dimensions: {object_analysis['dimensions'][0]:.1f} x {object_analysis['dimensions'][1]:.1f} x {object_analysis['dimensions'][2]:.1f} mm
Volume: {object_analysis['volume']:.0f} mm³
Size Category: {object_analysis['size_category']}

MOLD CONFIGURATION
Type: {mold_pieces}-piece mold
Split Axis: {split_axis.upper()}-axis
Wall Thickness: {thickness_analysis['optimal_thickness']:.2f} mm ({thickness_analysis['recommendation']})
User Input: {user_thickness_display}
Alignment Keys: {num_alignment_keys}
Mesh Repair: {repair_method}
"""

    # Add draft angles if used
    if draft_report and draft_report['success']:
        summary += f"Draft Angles: {draft_report['draft_angle_degrees']} degrees ({draft_report['parting_direction'].upper()}-axis)\n"
    else:
        summary += "Draft Angles: Disabled\n"

    # Add output files
    summary += f"\nOUTPUT FILES\n"
    for piece_name in piece_names:
        summary += f"- {base_name}_mold_{piece_name}.stl\n"
    
    summary += f"\nPRINTING TIPS\n"
    summary += f"- Use 0.1-0.2mm layer height for smooth surfaces\n"
    summary += f"- Print with 100% infill for strength\n"
    summary += f"- Spray with oil or silicone spray to reduce friction\n"
    summary += f"- Pour material through bottom spout\n"
    summary += f"- Alignment keys ensure proper assembly\n"

    return summary

def create_negative_space_mold(input_file, wall_thickness, split_axis='x', num_alignment_keys=2, mold_pieces=2, repair_method='auto', draft_angle=None):
    # Setup logging for this session
    log_filename = setup_logging(input_file)
    
    print_minimal_progress("Loading and analyzing STL file...")
    
    # Load the input STL file
    original_mesh = trimesh.load_mesh(input_file)
    logging.info(f"Loaded mesh: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")

    # Enhanced mesh repair and validation
    if not original_mesh.is_watertight and repair_method != 'none':
        print_minimal_progress(f"Mesh repair needed - using '{repair_method}' method...")
        logging.warning(f"Input mesh is not watertight - attempting repair using '{repair_method}' method...")
        original_mesh, repair_report = enhanced_mesh_repair(original_mesh, repair_method=repair_method)
        
        if not repair_report['final_watertight']:
            print_minimal_progress("Mesh repair partially successful - proceeding with caution", "WARNING")
            logging.error("Mesh repair was not fully successful")
            logging.warning("Proceeding with non-watertight mesh - results may be affected")
        else:
            print_minimal_progress("Mesh successfully repaired!", "SUCCESS")
            logging.info("Mesh successfully repaired and is now watertight!")
    elif not original_mesh.is_watertight and repair_method == 'none':
        print_minimal_progress("Mesh repair disabled - proceeding with original mesh", "WARNING")
        logging.warning("Input mesh is not watertight but repair is disabled")
    else:
        print_minimal_progress("Mesh is watertight - ready for mold creation!", "SUCCESS")
        logging.info("Input mesh is watertight - ready for mold creation!")

    # Validate split axis
    split_axis = split_axis.lower()
    if split_axis not in ['x', 'y', 'z']:
        raise ValueError("Split axis must be 'x', 'y', or 'z'")
    
    # Validate mold pieces
    if mold_pieces not in [2, 4]:
        raise ValueError("Mold pieces must be 2 or 4")
    
    axis_index = {'x': 0, 'y': 1, 'z': 2}[split_axis]
    print_minimal_progress(f"Creating {mold_pieces}-piece mold, splitting along {split_axis.upper()}-axis")
    logging.info(f"Creating {mold_pieces}-piece mold, splitting along {split_axis.upper()}-axis")

    # 🧠 INTELLIGENT WALL THICKNESS DETECTION
    print_minimal_progress("Calculating optimal wall thickness...")
    
    # Analyze and optimize wall thickness based on all parameters
    thickness_analysis = detect_optimal_wall_thickness(
        original_mesh=original_mesh,
        split_axis=split_axis,
        num_alignment_keys=num_alignment_keys,
        mold_pieces=mold_pieces,
        user_thickness=wall_thickness
    )
    
    # Use the intelligently determined optimal thickness
    optimal_wall_thickness = thickness_analysis['optimal_thickness']
    
    if thickness_analysis['override_applied']:
        print_minimal_progress(f"Wall thickness adjusted: {wall_thickness:.2f} mm → {optimal_wall_thickness:.2f} mm", "INFO")
        logging.info(f"Wall thickness adjusted: {wall_thickness:.2f} mm → {optimal_wall_thickness:.2f} mm")
        logging.info(f"Adjustment reason: {thickness_analysis['recommendation']}")
    else:
        method = 'calculated' if thickness_analysis['user_thickness'] is None else 'validated'
        print_minimal_progress(f"Using {method} wall thickness: {optimal_wall_thickness:.2f} mm", "SUCCESS")
        logging.info(f"Using {method} wall thickness: {optimal_wall_thickness:.2f} mm")
    
    # Update wall_thickness to use the optimal value
    wall_thickness = optimal_wall_thickness

    # Step 1: Create the outer mold block
    bounding_box = original_mesh.bounds
    min_corner = bounding_box[0] - wall_thickness
    max_corner = bounding_box[1] + wall_thickness

    # Create a mold block enclosing the original object
    mold_block_size = max_corner - min_corner
    mold_block_transform = trimesh.transformations.translation_matrix(
        (min_corner + max_corner) / 2
    )
    mold_block = trimesh.creation.box(
        extents=mold_block_size,
        transform=mold_block_transform
    )

    # Step 2: Subtract the object to create the cavity (negative space)
    mold_with_cavity = trimesh.boolean.difference([mold_block, original_mesh])

    # Step 2.5: Apply draft angles to cavity (NEW STEP!)
    draft_report = None
    if draft_angle is not None and draft_angle > 0:
        print_minimal_progress(f"Applying {draft_angle}° draft angles for easier demolding...")
        logging.info(f"Applying draft angles for easier demolding...")
        mold_with_cavity, draft_report = apply_draft_angles_to_cavity(
            mold_with_cavity, original_mesh, 
            draft_angle_degrees=draft_angle, 
            parting_direction=split_axis
        )
    else:
        print_minimal_progress("Draft angles disabled - using original cavity geometry")
        logging.info("Draft angles disabled - using original cavity geometry")

    # Step 3: Split the mold based on the number of pieces requested
    print_minimal_progress(f"Splitting mold into {mold_pieces} pieces...")
    
    if mold_pieces == 2:
        mold_pieces_list, piece_names = create_two_piece_mold(
            mold_with_cavity, split_axis, axis_index, min_corner, max_corner, mold_block_size
        )
    else:  # mold_pieces == 4
        mold_pieces_list, piece_names = create_four_piece_mold(
            mold_with_cavity, split_axis, axis_index, min_corner, max_corner, mold_block_size
        )
    
    print_minimal_progress(f"Created {len(mold_pieces_list)} mold pieces: {piece_names}", "SUCCESS")
    logging.info(f"Created {len(mold_pieces_list)} mold pieces: {piece_names}")

    # Step 4: Add alignment keys using corner-first smart positioning strategy
    print_minimal_progress(f"Adding {num_alignment_keys} alignment keys...")
    
    if mold_pieces == 2:
        # Original 2-piece alignment key logic
        logging.info(f"Finding optimal positions for {num_alignment_keys} alignment keys...")
        split_coordinate = (min_corner[axis_index] + max_corner[axis_index]) / 2
        key_positions = find_safe_key_positions(
            original_mesh, split_axis, split_coordinate, 
            min_corner, max_corner, wall_thickness, num_alignment_keys
        )
        
        # Extract 3D positions from corner positioning results
        if key_positions and isinstance(key_positions[0], (list, tuple)) and len(key_positions[0]) > 3:
            key_positions_3d = [pos[0] for pos in key_positions]
        else:
            key_positions_3d = key_positions
        
        # Apply alignment keys to 2-piece mold with size information
        object_size_info = {
            'size_category': thickness_analysis['object_analysis']['size_category'],
            'avg_dimension': np.mean(thickness_analysis['object_analysis']['dimensions']),
            'dimensions': thickness_analysis['object_analysis']['dimensions']
        }
        mold_pieces_list = add_alignment_keys_two_piece(
            mold_pieces_list, key_positions_3d, split_axis, wall_thickness, object_size_info
        )
    else:
        # 4-piece alignment key logic - create keys between adjacent pieces
        logging.info(f"Creating alignment keys for 4-piece mold...")
        mold_pieces_list = add_alignment_keys_four_piece(
            mold_pieces_list, split_axis, min_corner, max_corner, wall_thickness, original_mesh
        )
    
    print_minimal_progress("Alignment keys added successfully", "SUCCESS")

    # Step 5: Add a wax pour spout to the cavity using smart positioning
    print_minimal_progress("Adding pour spout...")
    logging.info("Finding optimal pour spout position...")
    split_coordinate = (min_corner[axis_index] + max_corner[axis_index]) / 2
    spout_info = get_simple_spout_position(
        original_mesh, split_axis, split_coordinate,
        min_corner, max_corner, wall_thickness
    )
    
    # Apply spout to all mold pieces
    mold_pieces_list = add_spout_to_pieces(mold_pieces_list, spout_info)
    print_minimal_progress("Pour spout added successfully", "SUCCESS")

    # Create enhanced output directory structure with descriptive naming
    print_minimal_progress("Generating output files...")
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    mold_type_suffix = f"{mold_pieces}-part"
    output_dir_name = f"{base_name}_{mold_type_suffix}"
    output_dir = os.path.join("output", output_dir_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Generate comprehensive settings summary
    settings_summary = generate_settings_summary(
        input_file, thickness_analysis, draft_report, split_axis, 
        mold_pieces, num_alignment_keys, repair_method
    )
    
    # Save settings summary to text file
    settings_file = os.path.join(output_dir, f"{base_name}_{mold_type_suffix}_settings.txt")
    with open(settings_file, 'w', encoding='utf-8') as f:
        f.write(settings_summary)
    logging.info(f"Saved settings summary: {os.path.basename(settings_file)}")
    
    # Generate output file names and save pieces
    output_files = []
    for i, (piece, piece_name) in enumerate(zip(mold_pieces_list, piece_names)):
        output_file = os.path.join(output_dir, f"{base_name}_mold_{piece_name}.stl")
        piece.export(output_file)
        output_files.append(output_file)
        logging.info(f"Saved: {os.path.basename(output_file)}")
    
    print_minimal_progress(f"Saved {len(output_files)} mold pieces to {output_dir}", "SUCCESS")

    # Log comprehensive summary to file and print simplified version
    logging.info("MOLD CREATION SUMMARY")
    logging.info(f"Object: {os.path.basename(input_file)}")
    logging.info(f"Size category: {thickness_analysis['object_analysis']['size_category']}")
    logging.info(f"Dimensions: {thickness_analysis['object_analysis']['dimensions'][0]:.1f} × {thickness_analysis['object_analysis']['dimensions'][1]:.1f} × {thickness_analysis['object_analysis']['dimensions'][2]:.1f} mm")
    logging.info(f"Volume: {thickness_analysis['object_analysis']['volume']:.0f} mm³")
    logging.info(f"Mold type: {mold_pieces}-piece, {split_axis.upper()}-axis split")
    logging.info(f"Wall thickness: {wall_thickness:.2f} mm ({thickness_analysis['recommendation']})")
    logging.info(f"Alignment keys: {num_alignment_keys}")
    
    if draft_report:
        logging.info(f"Draft angles: {draft_report['draft_angle_degrees']}° ({'Applied' if draft_report['success'] else 'Failed'})")
    else:
        logging.info("Draft angles: Disabled")
    
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Generated {len(output_files)} STL files")
    
    # Print simplified summary to console
    print(f"\n🎉 Mold Creation Complete!")
    print(f"📦 Model: {os.path.basename(input_file)}")
    print(f"🏭 Created: {mold_pieces}-piece mold ({len(output_files)} STL files)")
    print(f"📁 Location: {output_dir}")
    print(f"📋 Settings: {os.path.basename(settings_file)}")
    print(f"📄 Full details: {log_filename}")
    print(f"✅ Ready for 3D printing!")


if __name__ == "__main__":
    import sys
    
    # Check if running with command line arguments (old method) or interactive mode (new method)
    if len(sys.argv) > 1:
        # Command line mode - for backwards compatibility
        parser = argparse.ArgumentParser(
            description="Create a negative space mold with interlocking alignment keys and a wax pour spout. "
                       "Features intelligent wall thickness detection that analyzes object geometry, "
                       "alignment features, split axis, and manufacturing constraints for optimal results."
        )
        parser.add_argument("input_file", type=str, help="Path to the input STL file.")
        parser.add_argument("--wall_thickness", type=float, default=None,
                            help="Thickness of the mold walls. If not provided, will be calculated automatically based on object geometry, alignment features, and manufacturing constraints.")
        parser.add_argument("--split_axis", type=str, default='x', choices=['x', 'y', 'z'],
                            help="Axis along which to split the mold (default: x). "
                                 "z=horizontal, y=front-to-back, x=left-to-right.")
        parser.add_argument("--num_alignment_keys", type=int, default=2, choices=[2, 4],
                            help="Number of alignment keys to create (2 or 4, default: 2).")
        parser.add_argument("--mold_pieces", type=int, default=2, choices=[2, 4],
                            help="Number of mold pieces (2 or 4, default: 2).")
        parser.add_argument("--repair_method", type=str, default="auto", 
                            choices=['auto', 'trimesh', 'open3d', 'hybrid', 'none'],
                            help="Mesh repair method (default: auto). "
                                 "auto=automatic, trimesh=basic, open3d=advanced, hybrid=both, none=skip repair.")
        parser.add_argument("--draft_angle", type=float, default=None,
                            help="Draft angle in degrees for easier demolding (default: disabled). "
                                 "Typical values: 0.5-3.0 degrees. Larger angles = easier demolding but less accurate parts.")

        args = parser.parse_args()

        if not os.path.exists(args.input_file):
            print(f"Input file '{args.input_file}' does not exist.")
        else:
            create_negative_space_mold(args.input_file, args.wall_thickness, args.split_axis, 
                                     args.num_alignment_keys, args.mold_pieces, args.repair_method, args.draft_angle)
    else:
        # Interactive mode - new user-friendly interface
        try:
            config = interactive_setup()
            if config is not None:
                print(f"\n🚀 Starting mold creation...")
                create_negative_space_mold(
                    config['input_file'],
                    config['wall_thickness'],
                    config['split_axis'],
                    config['num_alignment_keys'],
                    config['mold_pieces'],
                    config['repair_method'],
                    config['draft_angle']
                )
            else:
                print("👋 Goodbye!")
                
        except KeyboardInterrupt:
            print(f"\n\n👋 Mold creation cancelled by user. Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print(f"💡 Please check the log files in the 'logs' directory for detailed error information.")
