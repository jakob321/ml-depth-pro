import numpy as np
import random
from PIL import Image
import depth_pro
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import os



class PyvistaPlotter:
    def __init__(self):
        pv.set_plot_theme("document")
        self.pl = pv.Plotter()
        self.pl.add_text("3D projection")
        self.camera_exists = False
        self.camera_center = np.array([1, 0, 0])
        self.focal_point = np.array([0, 0, 0])
        self.near_range = 0.1

    def create_camera(self, camera_center, focal_point, near_range, far_range):
        self.camera_exists = True
        self.camera_center = camera_center
        self.focal_point = focal_point
        self.near_range = near_range

        # Create a new camera
        camera = pv.Camera()
        camera.clipping_range = (near_range, far_range)
        camera.position = camera_center
        camera.focal_point = focal_point

        frustum = camera.view_frustum(1.0)
        line = pv.Line(camera_center, focal_point)

        self.pl.add_mesh(frustum, style="wireframe")  # Draw the frustum of the camera
        self.pl.add_mesh(
            line, color="b"
        )  # Draw a line from the camera to the focal point
        self.pl.add_point_labels(
            [
                camera_center,
            ],
            ["Camera Position"],
            margin=0,
            fill_shape=False,
            font_size=14,
            shape_color="white",
            point_color="red",
            text_color="black",
        )
        self.pl.add_arrows(np.array([1, 0, 0]), np.array([-1, 0, 0]), mag=0.2)

    def project_points(self, depth, np_img):
        if not self.camera_exists:
            print("Camera not created")
            return
        
        # Remove outliers
        if depth.max() > 9999:
            depth[depth > 9999] = 0
            dynamic_threshold = np.percentile(depth, 90)
            depth[depth > dynamic_threshold] = 0
        
        # Create plane of the image
        v = self.focal_point - self.camera_center  # direction vector
        v_norm = v / np.linalg.norm(v)
        distance = self.near_range
        Pc = (
            self.camera_center + distance * v_norm
        )  # Intersect point with plane of image

        # Create image plane and parametrize the plane using the two basis vectors
        w = (
            np.array([1, 0, 0])
            if v_norm[0] == 0 and v_norm[1] == 0
            else np.array([0, 0, 1])
        )
        u1 = np.cross(v_norm, w)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(v_norm, u1)
        u2 /= np.linalg.norm(u2)

        # Generate points on the image plane
        n_points_h = depth.shape[0]
        n_points_w = depth.shape[1]
        d = 0.001
        dist_w = depth.shape[1] * d
        dist_h = depth.shape[0] * d
        alphas = np.linspace(-dist_w, dist_w, n_points_w)  # Parameter for u1
        betas = np.linspace(-dist_h, dist_h, n_points_h)  # Parameter for u2
        alpha_grid, beta_grid = np.meshgrid(alphas, betas)
        plane_points = (
            Pc + np.outer(alpha_grid.ravel(), u1) + np.outer(beta_grid.ravel(), u2)
        )
        np_plane_points = np.array(plane_points)

        # Create vectors from camera center through generated "pixel"
        camera_center_repeated = np.tile(
            self.camera_center, (np_plane_points.shape[0], 1)
        )
        vectors_to_plane = np_plane_points - camera_center_repeated
        vector_magnitudes = np.linalg.norm(
            vectors_to_plane, axis=1, keepdims=True
        )  # Compute magnitudes
        normalized_vectors = vectors_to_plane / vector_magnitudes

        # Scale vectors with predicted depth
        desired_length = np.array(depth).reshape(depth.shape[1] * depth.shape[0], 1)
        points_at_distance = self.camera_center + normalized_vectors * desired_length

        # visualize the points
        np_img_flat = np.array(np_img).reshape(depth.shape[1] * depth.shape[0], 3)
        self.pl.add_points(
            points_at_distance,
            scalars=np_img_flat,
            render_points_as_spheres=True,
            rgb=True,
            point_size=5,
        )
        # pl.add_points(np_plane_points, color="blue", render_points_as_spheres=True, point_size=5)
        self.pl.remove_scalar_bar()

    def show(self):
        self.pl.show()


if __name__ == "__main__":
    # Load model and preprocessing transform
    print("Loading model...")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Visualize the depth map.
    plotter = PyvistaPlotter()
    plotter.create_camera(
        camera_center=np.array([1, 0, 0]),
        focal_point=np.array([0, 0, 0]),
        near_range=0.6,
        far_range=1.2,
    )
    

    # folder_path = "../kitty_dataset/image_02/data/"
    # image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
    # for image_path in image_paths[:5]:
    #     print(image_path)

        # Load and preprocess an image.
    image_path = "../kitty_dataset/image_02/data/0000000397.png"
    original_image = np.asarray(Image.open(image_path))
    # image_path = "../../../test_images/dark_and_light_forest.jpeg"
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    # Run inference.
    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"]  # Depth in [m].
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.


    print("done with model now starting visualization")

    plotter.project_points(depth, original_image)

    plotter.show()
    
    