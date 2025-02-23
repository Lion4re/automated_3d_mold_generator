#!/usr/bin/env python
"""
mold_generator.py

A command-line tool for generating 3D molds from input STL files using trimesh.
It includes enhanced mesh repair (using Open3D), draft angle application, hole processing,
and the ability to split the mold into halves or quarters.
"""

import trimesh
import numpy as np
import os
import logging
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def enhanced_mesh_repair(mesh):
    """
    Attempts to repair the input mesh using Open3D functionalities.
    This function converts the trimesh mesh to an Open3D TriangleMesh,
    performs various cleaning operations, and converts it back to trimesh.
    
    Parameters:
      mesh (trimesh.Trimesh): Input mesh.
    
    Returns:
      repaired_mesh (trimesh.Trimesh): The repaired mesh.
    """
    try:
        import open3d as o3d
    except ImportError:
        logging.error("Open3D is not installed. Please install it with 'pip install open3d'.")
        raise

    logging.info("Attempting enhanced mesh repair using Open3D.")
    # Convert trimesh to Open3D mesh:
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
    mesh_o3d.compute_vertex_normals()
    
    # Remove degenerate triangles and duplicated elements:
    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()
    
    # Convert back to trimesh:
    repaired_vertices = np.asarray(mesh_o3d.vertices)
    repaired_faces = np.asarray(mesh_o3d.triangles)
    repaired_mesh = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces, process=True)
    return repaired_mesh


def apply_draft(mesh, draft_angle):
    """
    Applies a draft (taper) to the input mesh.
    Each vertex is scaled in the x and y directions by a factor that increases
    linearly from the bottom (z_min) to the top (z_max) of the mesh.
    
    Parameters:
      mesh (trimesh.Trimesh): Input mesh.
      draft_angle (float): Draft angle in degrees.
      
    Returns:
      new_mesh (trimesh.Trimesh): Mesh with applied draft.
    """
    if draft_angle == 0:
        return mesh

    z_min = mesh.bounds[0][2]
    z_max = mesh.bounds[1][2]
    if z_max == z_min:
        return mesh

    angle_rad = np.deg2rad(draft_angle)
    new_vertices = mesh.vertices.copy()
    factors = 1.0 + ((new_vertices[:, 2] - z_min) / (z_max - z_min)) * np.tan(angle_rad)
    new_vertices[:, 0] *= factors
    new_vertices[:, 1] *= factors

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh.faces)
    return new_mesh


def create_mold_with_preserved_cavity(input_model_path, padding=0.1, hole_positions=['top'], split_mode='quarters', draft_angle=0.0):
    """
    Generates a mold with a preserved cavity from the input model.
    Optionally applies a draft angle and splits the mold into halves or quarters.
    """
    logging.info(f"Starting mold creation for model: {input_model_path}")
    mesh = trimesh.load_mesh(input_model_path)

    if not mesh.is_watertight:
        logging.warning("Input mesh is not watertight. Attempting to fill holes.")
        mesh.fill_holes()
        if not mesh.is_watertight:
            logging.warning("Mesh still not watertight after fill_holes. Attempting enhanced repair using Open3D.")
            mesh = enhanced_mesh_repair(mesh)
            if not mesh.is_watertight:
                logging.error("Mesh remains non-watertight after enhanced repair.")
                raise ValueError("Input mesh is not watertight even after enhanced repair.")
            else:
                logging.info("Mesh repaired using enhanced repair and is now watertight.")
    logging.info("Input mesh is watertight.")
    
    # Apply draft angle if specified
    if draft_angle != 0.0:
        logging.info(f"Applying a draft angle of {draft_angle} degrees to the cavity.")
        mesh = apply_draft(mesh, draft_angle)
    
    # Calculate bounding box of the mesh (cavity)
    mesh_bounds = mesh.bounds
    logging.info(f"Mesh bounds:\n{mesh_bounds}")

    # Create a slightly larger box (mold)
    mold_min = mesh_bounds[0] - padding
    mold_max = mesh_bounds[1] + padding
    mold_extents = mold_max - mold_min
    mold_center = (mold_min + mold_max) / 2
    logging.info(f"Mold extents: {mold_extents}")
    logging.info(f"Mold center: {mold_center}")

    mold_box = trimesh.creation.box(
        extents=mold_extents,
        transform=trimesh.transformations.translation_matrix(mold_center)
    )
    logging.info("Created mold box.")

    # Create the mold by subtracting the mesh from the box
    mold = mold_box.difference(mesh)
    logging.info("Created mold with cavity.")

    # Process holes if specified
    for position in hole_positions:
        logging.info(f"Processing hole at position: {position}")
        hole_radius = min(mold_extents[:2]) * 0.1
        
        if position == 'bottom':
            cavity_mid_z = (mesh_bounds[0][2] + mesh_bounds[1][2]) / 2
            hole_length = (cavity_mid_z - mold_min[2]) + padding
            
            hole_cylinder = trimesh.creation.cylinder(
                radius=hole_radius,
                height=hole_length,
                sections=64
            )
            
            hole_center = mold_center.copy()
            hole_center[2] = mold_min[2] + hole_length / 2
            
            hole_cylinder.apply_translation(hole_center - hole_cylinder.center_mass)
            mold = mold.difference(hole_cylinder)
            logging.info("Added bottom hole.")
        # Extend with additional positions as needed.

    # Split the mold according to split_mode
    parts = []
    if split_mode == 'halves':
        logging.info("Splitting the mold into halves along the X-axis.")
        left_box = trimesh.creation.box(
            extents=[mold_extents[0]/2, mold_extents[1] + padding, mold_extents[2] + 2*padding],
            transform=trimesh.transformations.translation_matrix([
                mold_center[0] - mold_extents[0]/4,
                mold_center[1],
                mold_center[2]
            ])
        )
        right_box = trimesh.creation.box(
            extents=[mold_extents[0]/2, mold_extents[1] + padding, mold_extents[2] + 2*padding],
            transform=trimesh.transformations.translation_matrix([
                mold_center[0] + mold_extents[0]/4,
                mold_center[1],
                mold_center[2]
            ])
        )
        left_part = mold.intersection(left_box)
        right_part = mold.intersection(right_box)
        parts = [left_part, right_part]
        logging.info("Created left and right halves.")
    elif split_mode == 'quarters':
        logging.info("Splitting the mold into quarters along X and Y axes.")
        extra = max(mold_extents) * 0.1
        x_center, y_center, _ = mold_center
        quarter_extents = [
            mold_extents[0] / 2 + extra,
            mold_extents[1] / 2 + extra,
            mold_extents[2] + 2 * extra
        ]
        fl_center = [x_center - mold_extents[0]/4, y_center - mold_extents[1]/4, mold_center[2]]
        fl_box = trimesh.creation.box(
            extents=quarter_extents,
            transform=trimesh.transformations.translation_matrix(fl_center)
        )
        fr_center = [x_center + mold_extents[0]/4, y_center - mold_extents[1]/4, mold_center[2]]
        fr_box = trimesh.creation.box(
            extents=quarter_extents,
            transform=trimesh.transformations.translation_matrix(fr_center)
        )
        bl_center = [x_center - mold_extents[0]/4, y_center + mold_extents[1]/4, mold_center[2]]
        bl_box = trimesh.creation.box(
            extents=quarter_extents,
            transform=trimesh.transformations.translation_matrix(bl_center)
        )
        br_center = [x_center + mold_extents[0]/4, y_center + mold_extents[1]/4, mold_center[2]]
        br_box = trimesh.creation.box(
            extents=quarter_extents,
            transform=trimesh.transformations.translation_matrix(br_center)
        )
        fl = mold.intersection(fl_box)
        fr = mold.intersection(fr_box)
        bl = mold.intersection(bl_box)
        br = mold.intersection(br_box)
        parts = [fl, fr, bl, br]
        logging.info("Created front-left, front-right, back-left, and back-right quarters.")
    else:
        logging.info("No splitting selected; returning full mold.")
        parts = [mold]
    
    return mold, parts


def process_single_model(file_path, output_dir, padding=0.1, hole_positions=['bottom'],
                         split_mode='quarters', visualize=False, draft_angle=0.0):
    """
    Processes a single model and exports the generated mold parts to the specified output directory.
    Optionally visualizes the full mold before exporting.
    """
    try:
        logging.info(f"Processing model: {file_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        mold, parts = create_mold_with_preserved_cavity(file_path, padding,
                                                        hole_positions, split_mode, draft_angle)
        
        if visualize:
            try:
                mold.show()
            except ModuleNotFoundError:
                logging.warning("Visualization not available because pyglet is not installed. Skipping visualization.")
        
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        logging.info("Exporting mold parts...")
        mold.export(os.path.join(output_dir, f'{base_filename}_complete_mold.stl'))
        for i, part in enumerate(parts):
            part.export(os.path.join(output_dir, f'{base_filename}_part_{i+1}.stl'))
        
        logging.info(f"Successfully exported all parts to {output_dir}")
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        raise


def process_models_in_directory(input_dir, output_dir, padding=0.1, hole_positions=['bottom'],
                                split_mode='quarters', visualize=False, draft_angle=0.0):
    """
    Processes all STL files in the input directory and exports their mold parts.
    """
    stl_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.stl')]
    for stl_file in tqdm(stl_files, desc="Processing STL files"):
        file_path = os.path.join(input_dir, stl_file)
        process_single_model(file_path, output_dir, padding, hole_positions,
                             split_mode, visualize, draft_angle)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a mold from a 3D model (STL) by preserving the cavity."
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to the input STL file or directory containing STL files."
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for the generated mold files."
    )
    parser.add_argument(
        "--padding", type=float, default=0.1,
        help="Padding added around the input mesh (default: 0.1)."
    )
    parser.add_argument(
        "--hole_positions", type=str, nargs="+", default=["bottom"],
        help="Positions to add holes (e.g., bottom, top, left, right). Default is 'bottom'."
    )
    parser.add_argument(
        "--split_mode", type=str, choices=["halves", "quarters"], default="quarters",
        help="How to split the mold (halves or quarters). Default is quarters."
    )
    parser.add_argument(
        "--draft_angle", type=float, default=0.0,
        help="Draft angle in degrees to apply to the cavity (default: 0.0)."
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize the mold before exporting."
    )

    args = parser.parse_args()

    # If input is a directory, process all STL files
    if os.path.isdir(args.input):
        process_models_in_directory(
            input_dir=args.input,
            output_dir=args.output,
            padding=args.padding,
            hole_positions=args.hole_positions,
            split_mode=args.split_mode,
            visualize=args.visualize,
            draft_angle=args.draft_angle
        )
    else:
        process_single_model(
            file_path=args.input,
            output_dir=args.output,
            padding=args.padding,
            hole_positions=args.hole_positions,
            split_mode=args.split_mode,
            visualize=args.visualize,
            draft_angle=args.draft_angle
        )


if __name__ == "__main__":
    main()
