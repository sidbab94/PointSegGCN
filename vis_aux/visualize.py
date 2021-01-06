import os
from yaml import safe_load
import numpy as np

from vispy.scene import visuals, SceneCanvas
from vispy.io import read_png, load_data_file
import vispy
from .laserscan import SemLaserScan
from preproc_utils.dataprep import get_labels, read_bin_velodyne
import open3d as o3d

class ScanVis:
    def __init__(self, config, scan_file, orig_labels, pred_labels=None):
        self.scan_file = scan_file
        self.orig_labels = orig_labels
        self.pred_labels = pred_labels
        vispy.use(app='PyQt4')


        color_dict = config["color_map"]
        nclasses = len(color_dict)
        self.scan_obj = SemLaserScan(nclasses, color_dict, project=True)

        self.canvas = SceneCanvas(keys='interactive', show=True)
        self.canvas.events.key_press.connect(self.key_press)
        # self.canvas.events.draw.connect(self.draw)
        self.grid = self.canvas.central_widget.add_grid()

        print("Using semantics in visualizer")
        self.orig_sem_view = vispy.scene.widgets.ViewBox(
            border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.orig_sem_view, 0, 0)
        self.orig_sem_vis = visuals.Markers()
        self.orig_sem_view.camera = 'turntable'
        # self.orig_sem_view.transform = MatrixTransform()
        # self.orig_sem_view.transform.rotate(90, axis=(1, 0, 0))
        self.orig_sem_view.add(self.orig_sem_vis)
        visuals.XYZAxis(parent=self.orig_sem_view.scene)


        if self.pred_labels is not None:
            self.pred_sem_view = vispy.scene.widgets.ViewBox(
                border_color='red', parent=self.canvas.scene)
            self.grid.add_widget(self.pred_sem_view, 1, 0)
            self.pred_sem_vis = visuals.Markers()
            self.pred_sem_view.camera = 'turntable'
            self.pred_sem_view.add(self.pred_sem_vis)
            visuals.XYZAxis(parent=self.pred_sem_view.scene)

        self.update_scan()

    def update_scan(self):
        # first open data
        self.scan_obj.open_scan(self.scan_file)

        # self.scan_obj.open_label(self.label_file)
        self.scan_obj.colorize(self.orig_labels)
        self.orig_sem_vis.set_data(self.scan_obj.points,
                              face_color=self.scan_obj.sem_label_color[..., ::-1],
                              edge_color=self.scan_obj.sem_label_color[..., ::-1],
                              size=1)

        if self.pred_labels is not None:
            self.scan_obj.colorize(self.pred_labels)
            self.pred_sem_vis.set_data(self.scan_obj.points,
                                  face_color=self.scan_obj.sem_label_color[..., ::-1],
                                  edge_color=self.scan_obj.sem_label_color[..., ::-1],
                                  size=1)

    def key_press(self, event):
        self.canvas.events.key_press.block()
        if event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()

    def img_velo_vis(self, image_file):
        if self.pred_labels is not None:
            raise Exception('Second view-box already occupied')

        rgb = read_png(load_data_file(image_file))
        self.img_grid = self.canvas.central_widget.add_grid()
        self.img_view = vispy.scene.widgets.ViewBox(
            border_color='blue', parent=self.canvas.scene)

        self.img_grid.add_widget(self.img_view, 1, 0)
        self.img_vis = visuals.Image(data=rgb, cmap='viridis')

        self.img_view.add(self.img_vis)
        self.img_view.camera = vispy.scene.PanZoomCamera()
        self.img_view.camera.set_range(margin=0.0)
        self.img_view.camera.flip = (0, 1, 0)
        visuals.XYZAxis(parent=self.img_view.scene)


class o3d_vis:
    def __init__(self, scan_file, rgb_data, lidar_to_camera=False):
        self.pc = scan_file

        if isinstance(self.pc, str):
            self.pc = read_bin_velodyne(scan_file)

        self.pc_obj = o3d.geometry.PointCloud()
        self.pc_obj.points = o3d.utility.Vector3dVector(self.pc[:, 0:3])

        if lidar_to_camera:
            self.pc_obj.colors = o3d.utility.Vector3dVector(rgb_data)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pc_obj)

        self.vis.run()
        self.vis.destroy_window()

    def rotate_view(self):
        ctr = self.vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    def transform(self):
        self.pc_obj.transform(self.tr)
        self.pc_obj.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    def get_camera_params(self, calib, img_size):

        self.calib = calib
        self.h, self.w = img_size

        K = self.calib['P2'].reshape((3, 4))
        ext = np.vstack((self.calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))

        f_x, f_y = K[0, 0] * self.w, K[1, 1] * self.w
        c_x, c_y = self.h/2 - 0.5, self.w/2 - 0.5

        int = o3d.camera.PinholeCameraIntrinsic()
        int.set_intrinsics(width=self.w, height=self.h,
                           fx=f_x, fy=f_y, cx=c_x, cy=c_y)

        self.camera_params = o3d.camera.PinholeCameraParameters()
        self.camera_params.extrinsic = ext
        self.camera_params.intrinsic = int




def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 2.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)



if __name__ == '__main__':
    BASE_DIR = 'D:/SemanticKITTI/dataset/sequences'
    pc = os.path.join(BASE_DIR, '08', 'velodyne', '000000.bin')
    label = os.path.join(BASE_DIR, '08', 'labels', '000000.label')
    CFG = safe_load(open('../config/semantic-kitti.yaml', 'r'))

    x = read_bin_velodyne(pc)
    y = get_labels(label, CFG)



    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)

