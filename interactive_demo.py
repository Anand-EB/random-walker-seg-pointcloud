#!/usr/bin/env python3
"""Minimal viser-based point cloud viewer for .ply/.las/.laz files."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Tuple

import numpy as np

try:
    from plyfile import PlyData
except ImportError:  # pragma: no cover - instructions-only path
    PlyData = None

try:
    import laspy
except ImportError:  # pragma: no cover - instructions-only path
    laspy = None

try:
    from pc_rwalker import random_walker_segmentation_gc, random_walker_segmentation
except ImportError:  # pragma: no cover - instructions-only path
    random_walker_segmentation_gc = None
    random_walker_segmentation = None

import viser


def _ensure_dependency(name: str, module: object | None) -> None:
    if module is None:
        raise SystemExit(
            f"The '{name}' package is required for this file type. "
            f"Install it with `pip install {name}`."
        )


def _load_ply_points(ply_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return (points, colors) from a PLY file."""
    _ensure_dependency("plyfile", PlyData)
    ply = PlyData.read(ply_path)

    if "vertex" not in ply:
        raise ValueError(f"{ply_path} does not contain a 'vertex' element.")

    vertex = ply["vertex"].data
    coord_fields = ("x", "y", "z")

    if not all(field in vertex.dtype.names for field in coord_fields):
        raise ValueError(f"{ply_path} is missing one of {coord_fields}.")

    points = np.column_stack([vertex[field] for field in coord_fields]).astype(
        np.float32, copy=False
    )

    colors = None
    rgb_fields = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    ]
    for fields in rgb_fields:
        if all(field in vertex.dtype.names for field in fields):
            colors = (
                np.column_stack([vertex[field] for field in fields])
                .astype(np.float32, copy=False)
                / 255.0
            )
            break

    return points, colors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a .ply/.las/.laz point cloud and visualize it with viser."
    )
    parser.add_argument(
        "ply_path",
        type=pathlib.Path,
        help="Path to the .ply file to visualize.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.9,
        help="Size of each rendered point in meters (default: 0.05).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the viser server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the viser server (default: 8080).",
    )
    return parser.parse_args()


def _load_las_points(las_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray | None]:
    _ensure_dependency("laspy[laszip]", laspy)
    las = laspy.read(las_path)
    points = (
        np.column_stack((las.x, las.y, las.z))
        .astype(np.float32, copy=False)
    )

    colors = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # LAS colors are typically uint16 0-65535.
        colors = (
            np.column_stack((las.red, las.green, las.blue))
            .astype(np.float32, copy=False)
            / 65535.0
        )

    return points, colors


def _load_points(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _load_ply_points(path)
    if suffix in {".las", ".laz"}:
        return _load_las_points(path)
    raise ValueError(f"Unsupported file extension '{suffix}'. Use .ply, .las, or .laz.")


def main() -> None:
    args = _parse_args()
    if not args.ply_path.exists():
        sys.exit(f"File '{args.ply_path}' not found.")

    points, colors = _load_points(args.ply_path)
    if points.size == 0:
        sys.exit(f"No vertices found in '{args.ply_path}'.")

    # Center the point cloud
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Calculate bounding box for camera positioning
    bbox_min = np.min(centered_points, axis=0)
    bbox_max = np.max(centered_points, axis=0)
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    print(f"Point cloud stats:")
    print(f"  Points: {points.shape[0]}")
    print(f"  Centroid: {centroid}")
    print(f"  Bounding box size: {bbox_size:.3f}")
    print(f"  Min: {bbox_min}")
    print(f"  Max: {bbox_max}")

    # Store original colors for restoration
    original_colors = colors.copy() if colors is not None else None
    current_colors = colors.copy() if colors is not None else np.ones((len(centered_points), 3), dtype=np.float32) * 0.5
    
    # Track selected point indices and markers
    selected_point_indices = set()
    selected_markers = {}
    
    # Track segmentation results
    segment_labels = None
    segment_colors_map = None

    server = viser.ViserServer(host=args.host, port=args.port)
    
    # Add the main point cloud
    point_cloud_handle = server.scene.add_point_cloud(
        name=args.ply_path.stem or "point_cloud",
        points=centered_points,
        colors=current_colors,
        point_size=args.point_size,
    )

    # Set up camera to view the point cloud
    # Position camera at a distance proportional to bbox size
    camera_distance = bbox_size * 1.5
    
    # Handler for client connections
    @server.on_client_connect
    def handle_new_client(client: viser.ClientHandle) -> None:
        client.camera.position = (camera_distance, camera_distance, camera_distance)
        client.camera.look_at = (0.0, 0.0, 0.0)
        
        # Add button to enable click selection (multi-select mode)
        click_button = client.gui.add_button(
            "Select Points (Click Multiple)",
            icon=viser.Icon.POINTER,
        )
        
        # Add button to clear all selections
        clear_button = client.gui.add_button(
            "Clear Selections",
            icon=viser.Icon.X,
        )
        
        @click_button.on_click
        def _(_) -> None:
            nonlocal selected_point_indices, selected_markers, current_colors
            
            click_button.disabled = True
            
            @client.scene.on_pointer_event(event_type="click")
            def handle_click(event: viser.ScenePointerEvent) -> None:
                nonlocal selected_point_indices, selected_markers, current_colors
                
                # Get ray origin and direction
                ray_origin = np.array(event.ray_origin)
                ray_direction = np.array(event.ray_direction)
                
                # Find the point closest to the ray
                # Calculate distance from each point to the ray
                # Using point-to-line distance formula
                point_to_origin = centered_points - ray_origin
                projection_length = np.dot(point_to_origin, ray_direction)
                projection = ray_origin + projection_length[:, None] * ray_direction
                distances = np.linalg.norm(centered_points - projection, axis=1)
                
                nearest_idx = np.argmin(distances)
                
                # Only select if ray is reasonably close to a point
                if distances[nearest_idx] < args.point_size * 20:
                    # Toggle selection - if already selected, deselect it
                    if nearest_idx in selected_point_indices:
                        # Deselect point
                        selected_point_indices.remove(nearest_idx)
                        
                        # Restore original color
                        if original_colors is not None:
                            current_colors[nearest_idx] = original_colors[nearest_idx]
                        else:
                            current_colors[nearest_idx] = [0.5, 0.5, 0.5]
                        
                        # Remove marker
                        if nearest_idx in selected_markers:
                            selected_markers[nearest_idx].remove()
                            del selected_markers[nearest_idx]
                        
                        print(f"Deselected point {nearest_idx}")
                    else:
                        # Select new point
                        selected_point_indices.add(nearest_idx)
                        
                        # Highlight point in red
                        current_colors[nearest_idx] = [1.0, 0.0, 0.0]
                        
                        # Add a sphere marker at selected point
                        marker = server.scene.add_icosphere(
                            name=f"selected_point_marker_{nearest_idx}",
                            radius=args.point_size * 2,
                            color=(255, 0, 0),
                            position=tuple(centered_points[nearest_idx]),
                        )
                        selected_markers[nearest_idx] = marker
                        
                        print(f"Selected point {nearest_idx}: position={centered_points[nearest_idx]} (Total: {len(selected_point_indices)} points)")
                    
                    # Update point cloud colors
                    point_cloud_handle.colors = current_colors
                
                # Don't remove callback - keep clicking enabled for multiple selections
            
            @client.scene.on_pointer_callback_removed
            def _() -> None:
                click_button.disabled = False
        
        @clear_button.on_click
        def _(_) -> None:
            nonlocal selected_point_indices, selected_markers, current_colors
            
            # Restore all colors
            for idx in list(selected_point_indices):
                if original_colors is not None:
                    current_colors[idx] = original_colors[idx]
                else:
                    current_colors[idx] = [0.5, 0.5, 0.5]
            
            # Remove all markers
            for marker in selected_markers.values():
                marker.remove()
            
            selected_point_indices.clear()
            selected_markers.clear()
            
            # Update point cloud colors
            point_cloud_handle.colors = current_colors
            
            # Stop selection mode if active
            client.scene.remove_pointer_callback()
            
            print(f"Cleared all selections")
        
        # Add segmentation controls
        with client.gui.add_folder("Segmentation"):
            algorithm_dropdown = client.gui.add_dropdown(
                "Algorithm",
                options=["Geometry-Central (GC)", "Legacy"],
                initial_value="Geometry-Central (GC)",
                hint="Choose segmentation algorithm"
            )
            
            n_neighbors_slider = client.gui.add_slider(
                "N Neighbors",
                min=5,
                max=100,
                step=5,
                initial_value=30,
                hint="Number of nearest neighbors for segmentation"
            )
            
            pc_segment_button = client.gui.add_button(
                "Segment Point Cloud",
                hint="Segment the point cloud using selected seed points"
            )
            
            show_labels_checkbox = client.gui.add_checkbox(
                "Show Segment Labels",
                initial_value=False,
                hint="Toggle between original colors and segment labels"
            )
            
            @pc_segment_button.on_click
            def pc_segment(_) -> None:
                nonlocal current_colors, segment_labels, segment_colors_map
                
                if random_walker_segmentation_gc is None and random_walker_segmentation is None:
                    client.gui.add_markdown("**Error:** pc_rwalker module not installed!")
                    print("Error: pc_rwalker module not available. Please install it first.")
                    return
                
                if len(selected_point_indices) == 0:
                    client.gui.add_markdown("**Warning:** No seed points selected!")
                    print("Warning: Please select at least one seed point before running segmentation.")
                    return
                
                # Determine which algorithm to use
                use_gc = algorithm_dropdown.value == "Geometry-Central (GC)"
                if use_gc and random_walker_segmentation_gc is None:
                    print("Warning: Geometry-Central algorithm not available. Falling back to Legacy.")
                    use_gc = False
                
                if not use_gc and random_walker_segmentation is None:
                    print("Error: Legacy algorithm not available.")
                    return
                
                pc_segment_button.disabled = True
                algorithm_name = "Geometry-Central" if use_gc else "Legacy"
                print(f"\nRunning {algorithm_name} segmentation with {len(selected_point_indices)} seed points...")
                print(f"  N neighbors: {n_neighbors_slider.value}")
                
                try:
                    # Prepare seed indices - each selected point is a separate seed group
                    seed_indices = [[idx] for idx in selected_point_indices]
                    
                    # Run segmentation on centered points
                    start_time = time.time()
                    if use_gc:
                        segment_labels = random_walker_segmentation_gc(
                            centered_points,
                            seed_indices,
                            n_neighbors=n_neighbors_slider.value,
                            return_flat=True
                        )
                    else:
                        segment_labels = random_walker_segmentation(
                            centered_points,
                            seed_indices,
                            n_neighbors=n_neighbors_slider.value,
                            return_flat=True,
                            n_proc=1
                        )
                    elapsed = time.time() - start_time
                    
                    print(f"{algorithm_name} segmentation completed in {elapsed:.2f} seconds")
                    print(f"  Segments: {len(np.unique(segment_labels))}")
                    
                    # Generate distinct colors for each segment
                    n_segments = len(seed_indices)
                    segment_colors = np.array([
                        [1.0, 0.0, 0.0],  # Red
                        [0.0, 1.0, 0.0],  # Green
                        [0.0, 0.0, 1.0],  # Blue
                        [1.0, 1.0, 0.0],  # Yellow
                        [1.0, 0.0, 1.0],  # Magenta
                        [0.0, 1.0, 1.0],  # Cyan
                        [1.0, 0.5, 0.0],  # Orange
                        [0.5, 0.0, 1.0],  # Purple
                    ])
                    
                    # If more segments than predefined colors, generate random colors
                    if n_segments > len(segment_colors):
                        extra_colors = np.random.rand(n_segments - len(segment_colors), 3)
                        segment_colors = np.vstack([segment_colors, extra_colors])
                    
                    # Store segment color mapping
                    segment_colors_map = np.zeros((len(centered_points), 3), dtype=np.float32)
                    for i in range(n_segments):
                        mask = segment_labels == i
                        segment_colors_map[mask] = segment_colors[i]
                    
                    # Keep seed points highlighted in their original red
                    for idx in selected_point_indices:
                        segment_colors_map[idx] = [1.0, 0.0, 0.0]
                    
                    # Enable the checkbox, set it to true, and show labels
                    show_labels_checkbox.disabled = False
                    show_labels_checkbox.value = True
                    current_colors = segment_colors_map.copy()
                    point_cloud_handle.colors = current_colors
                    
                    print("Segmentation visualization updated!")
                    
                except Exception as e:
                    print(f"Error during segmentation: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    pc_segment_button.disabled = False
            
            @show_labels_checkbox.on_update
            def _(_) -> None:
                nonlocal current_colors
                
                if segment_labels is None or segment_colors_map is None:
                    show_labels_checkbox.value = False
                    show_labels_checkbox.disabled = True
                    print("Run segmentation first before toggling labels.")
                    return
                
                if show_labels_checkbox.value:
                    # Show segment labels
                    current_colors = segment_colors_map.copy()
                    # Keep seed points highlighted
                    for idx in selected_point_indices:
                        current_colors[idx] = [1.0, 0.0, 0.0]
                    point_cloud_handle.colors = current_colors
                    print("Showing segment labels")
                else:
                    # Show original RGB colors
                    if original_colors is not None:
                        current_colors = original_colors.copy()
                        print("Showing original RGB colors")
                    else:
                        current_colors = np.ones((len(centered_points), 3), dtype=np.float32) * 0.7
                        print("Showing uniform color (no original RGB)")
                    
                    # Keep seed points highlighted
                    for idx in selected_point_indices:
                        current_colors[idx] = [1.0, 0.0, 0.0]
                    point_cloud_handle.colors = current_colors

    # Print proper URL for browser access
    if args.host == "0.0.0.0":
        browser_url = f"http://localhost:{args.port}"
    else:
        browser_url = f"http://{args.host}:{args.port}"
    
    print(f"\nViser server started!")
    print(f"Open {browser_url} in a browser to view the cloud.")
    print(f"\nWorkflow:")
    print(f"  1. Click 'Select Points (Click Multiple)' and select seed points")
    print(f"  2. Adjust 'N Neighbors' parameter if needed")
    print(f"  3. Click 'Segment Point Cloud' to segment the point cloud")
    print(f"  4. Use 'Clear Selections' to reset and try again")
    print(f"\nPress Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping viewer...")


if __name__ == "__main__":
    main()

