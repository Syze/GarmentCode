import os
import platform
if platform.system() == 'Linux':
    os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np
import trimesh
import pyrender
from PIL import Image

from pygarment.meshgen.sim_config import PathCofig


def rotate_matrix_y(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle), 0],
        [0, 1, 0, 0],
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def rotate_matrix_x(matrix, angle_deg):
    rotation_angle = angle_deg * (np.pi / 180)

    # Define the rotation matrix for 180-degree rotation around the y-axis
    rotation_matrix = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [0, np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 0, 1]
    ])

    # Apply the rotation to the mesh vertices
    rot_matrix = np.dot(rotation_matrix, matrix)
    return rot_matrix

def get_bounding_box_edges(mesh):
    # Calculate the bounding box of the mesh
    min_coords = mesh.bounds[0]
    max_coords = mesh.bounds[1]

    # Compute the corner points of the bounding box
    corners = [
        min_coords,
        [max_coords[0], min_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], min_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
        max_coords
    ]

    return corners

def create_camera(pyrender, pyrender_body_mesh, scene, side, camera_location=None):

    # Create a camera
    y_fov = np.pi / 6. 
    camera = pyrender.PerspectiveCamera(yfov=y_fov)
    

    if camera_location is None:
        # Evaluate w.r.t. body

        fov = 50  # Set your desired field of view in degrees 

        # # Calculate the bounding box center of the mesh
        bounding_box_center = pyrender_body_mesh.bounds.mean(axis=0)

        # Calculate the diagonal length of the bounding box
        diagonal_length = np.linalg.norm(pyrender_body_mesh.bounds[1] - pyrender_body_mesh.bounds[0])

        # Calculate the distance of the camera from the object based on the diagonal length
        distance = 1.5 * diagonal_length / (2 * np.tan(np.radians(fov / 2)))

        camera_location = bounding_box_center
        camera_location[-1] += distance

    # Calculate the camera pose
    camera_pose = np.array([
        [1.0, 0.0, 0.0, camera_location[0]],
        [0.0, 1.0, 0.0, camera_location[1]],
        [0.0, 0.0, 1.0, camera_location[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])

    camera_pose = rotate_matrix_x(camera_pose, -15)
    camera_pose = rotate_matrix_y(camera_pose, 20)
    if side == 'back':
        camera_pose = rotate_matrix_y(camera_pose, 180)

    # Set camera's pose in the scene
    scene.add(camera, pose=camera_pose)

def create_lights(scene, intensity=30.0):
    light_positions = [
        np.array([1.60614, 1.5341, 1.23701]),
        np.array([1.31844, 1.92831, -2.52238]),
        np.array([-2.80522, 1.2594, 2.34624]),
        np.array([0.160261, 1.81789, 3.52215]),
        np.array([-2.65752, 1.41194, -1.26328])
    ]
    light_colors = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

    # Add lights to the scene
    for i in range(5):
        light = pyrender.PointLight(color=light_colors[i], intensity=intensity)
        light_pose = np.eye(4)
        light_pose[:3, 3] = light_positions[i]
        scene.add(light, pose=light_pose)

def render(
        pyrender_garm_mesh, pyrender_body_mesh, 
        side, 
        paths: PathCofig, 
        render_props=None
    ):
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 1080, 1080
    # Create a pyrender scene
    scene = pyrender.Scene(bg_color=(1., 1., 1., 0.))  # Transparent!
    
    # Create a pyrender mesh object from the trimesh object
    # Add the mesh to the scene
    scene.add(pyrender_garm_mesh)
    scene.add(pyrender_body_mesh)

    camera_location=render_props['front_camera_location'] if 'front_camera_location' in render_props else None
    create_camera(
        pyrender, pyrender_body_mesh, scene, side,
        camera_location=camera_location
    )

    create_lights(scene, intensity=80.)

    # Create a renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)

    # Render the scene
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    image = Image.fromarray(color)
    image.save(paths.render_path(side), "PNG")

def load_meshes(paths:PathCofig, body_v, body_f):
    # Load body mesh
    body_mesh = trimesh.Trimesh(body_v, body_f)
    body_mesh.vertices = body_mesh.vertices / 100
    # Color body mesh
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.5, 0.5, 0.5, 1.0),  # Medium gray body
        metallicFactor=0.658,
        roughnessFactor=0.5
    )
    pyrender_body_mesh = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)


    #Load garment mesh
    garm_mesh = trimesh.load_mesh(str(paths.g_sim))  # NOTE: Includes the texture
    garm_mesh.vertices = garm_mesh.vertices / 100   # scale to m

    # Material adjustments
    material = garm_mesh.visual.material.to_pbr()
    material.baseColorFactor = [1., 1., 1., 1.]
    material.doubleSided = True  # color both face sides  
    # NOTE remove transparency -- add white background just in case
    white_back = Image.new('RGBA', material.baseColorTexture.size, color=(255, 255, 255, 255))
    white_back.paste(material.baseColorTexture)
    material.baseColorTexture = white_back.convert('RGB')  

    garm_mesh.visual.material = material

    pyrender_garm_mesh = pyrender.Mesh.from_trimesh(garm_mesh, smooth=True) 
    
    return pyrender_garm_mesh, pyrender_body_mesh

def render_frame_to_array(garment_verts, garment_faces, body_v, body_f, render_props):
    """Render garment + body from in-memory vertices and return as numpy RGBA array.

    Used for capturing simulation frames for video.
    """
    if render_props and 'resolution' in render_props:
        view_width, view_height = render_props['resolution']
    else:
        view_width, view_height = 540, 540

    # Body mesh
    body_mesh = trimesh.Trimesh(body_v / 100, body_f)
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.5, 0.5, 0.5, 1.0),  # Medium gray body
        metallicFactor=0.658,
        roughnessFactor=0.5
    )
    pyrender_body = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)

    # Garment mesh (solid color, no texture needed for video frames)
    garm_mesh = trimesh.Trimesh(garment_verts / 100, garment_faces)
    garm_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.12, 0.20, 0.38, 1.0),  # Dark navy garment
        metallicFactor=0.1,
        roughnessFactor=0.7,
        doubleSided=True
    )
    pyrender_garm = pyrender.Mesh.from_trimesh(garm_mesh, material=garm_material, smooth=True)

    # Scene
    scene = pyrender.Scene(bg_color=(0.85, 0.85, 0.85, 1.))
    scene.add(pyrender_garm)
    scene.add(pyrender_body)

    camera_location = render_props.get('front_camera_location') if render_props else None
    create_camera(pyrender, pyrender_body, scene, 'front', camera_location=camera_location)
    create_lights(scene, intensity=80.)

    renderer = pyrender.OffscreenRenderer(viewport_width=view_width, viewport_height=view_height)
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    return color


class FrameRenderer:
    """Reusable offscreen renderer for simulation video frames.

    render_frame_to_array() rebuilds everything per frame -- body mesh, both
    materials, scene, camera, lights, and a fresh OffscreenRenderer (an EGL
    context) that it then destroys. Only the garment geometry actually changes
    between frames, so this class builds the static half once and swaps just
    the garment node, which is where nearly all of the per-frame cost went.

    Owns a GL context: construct and use it on one thread only.
    """

    def __init__(self, body_v, body_f, render_props, resolution=None):
        if resolution is None:
            resolution = (render_props or {}).get('resolution', (540, 540))
        self.width, self.height = int(resolution[0]), int(resolution[1])

        body_mesh = trimesh.Trimesh(body_v / 100, body_f)
        body_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.5, 0.5, 0.5, 1.0),
            metallicFactor=0.658, roughnessFactor=0.5)
        pyrender_body = pyrender.Mesh.from_trimesh(body_mesh, material=body_material)

        self.garm_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.12, 0.20, 0.38, 1.0),
            metallicFactor=0.1, roughnessFactor=0.7, doubleSided=True)

        self.scene = pyrender.Scene(bg_color=(0.85, 0.85, 0.85, 1.))
        self.scene.add(pyrender_body)
        create_camera(pyrender, pyrender_body, self.scene, 'front',
                      camera_location=(render_props or {}).get('front_camera_location'))
        create_lights(self.scene, intensity=80.)

        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width, viewport_height=self.height)
        self.garm_node = None

    def render(self, garment_verts, garment_faces):
        """Swap in this frame's garment geometry and return an RGBA array."""
        if self.garm_node is not None:
            self.scene.remove_node(self.garm_node)
        mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(garment_verts / 100, garment_faces),
            material=self.garm_material, smooth=True)
        self.garm_node = self.scene.add(mesh)
        color, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        return color

    def close(self):
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None


def render_images(paths: PathCofig, body_v, body_f, render_props):

    pyrender_garm_mesh, pyrender_body_mesh = load_meshes(paths, body_v, body_f)

    for side in render_props['sides']:
        render(pyrender_garm_mesh, pyrender_body_mesh, side, paths, render_props)


