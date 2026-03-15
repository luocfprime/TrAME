import pickle
import socket
import struct
import threading

import math
import numpy as np
import viser
import viser.transforms as tf

from gaussiansplatting.scene.cameras import Simple_Camera


def stream_send(conn, serialized_data):
    message_size = struct.pack("=L", len(serialized_data))

    try:
        conn.sendall(message_size + serialized_data)
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.close()


def stream_recv(conn):
    message_size = conn.recv(struct.calcsize("=L"))
    message_size = struct.unpack("=L", message_size)[0]

    data = b""
    while len(data) < message_size:
        packet = conn.recv(message_size - len(data))
        if not packet:
            break
        data += packet

    return data


class RendererServerWorkingThread(threading.Thread):
    def __init__(self, conn, render_fn):
        super().__init__()
        self.conn = conn
        self.render_fn = render_fn

    def run(self):
        while True:
            try:
                data = stream_recv(self.conn)

                camera = pickle.loads(data)
                # render
                outputs = self.render_fn(camera)
                stream_send(self.conn, pickle.dumps(outputs))
            except Exception as e:
                print(e)
                break
        print("Connection closed")
        self.conn.close()


class RendererServer:
    def __init__(self, renderer_ip, renderer_port, render_fn):
        self.render_fn = render_fn

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((renderer_ip, renderer_port))
        self.sock.listen(1)
        self.sock.setblocking(False)  # non-blocking

    def start(self):
        """
        Start the server in a non-blocking way.
        """

        def connection_handler():
            while True:
                try:
                    conn, addr = self.sock.accept()
                    print(f"Connected by {addr}")
                    conn.setblocking(True)
                    worker = RendererServerWorkingThread(conn, self.render_fn)
                    worker.start()
                except BlockingIOError:
                    continue

        print(f"Renderer server started at {self.sock.getsockname()}")

        master_thread = threading.Thread(target=connection_handler)
        master_thread.daemon = True
        master_thread.start()


class ViserViewer:
    def __init__(self, viewer_port):
        self.port = viewer_port

        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button(
            label="Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )

        self.need_update = False

        self.FoV_slider = self.server.add_gui_slider(
            "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
        )

        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

    def set_renderer(self, renderer_ip, renderer_port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((renderer_ip, renderer_port))
        print("Connected!")

    def get_render(self, camera: Simple_Camera):
        """

        Args:

        Returns:
            ndarray, float32
        """
        stream_send(self.sock, pickle.dumps(camera))
        data = stream_recv(self.sock)
        return pickle.loads(data)

    def camera_to_simple_camera(self, viser_cam):
        aspect = viser_cam.aspect

        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position

        fovy = viser_cam.fov

        fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)

        width = int(self.resolution_slider.value)
        height = int(width / aspect)

        print(f"Camera info: T: {T}, T.norm: {np.sqrt(T[0] ** 2 + T[1] ** 2 + T[2])}, fovx: {fovx}, fovy: {fovy}")

        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def update(self):
        if self.need_update:
            for client in self.server.get_clients().values():
                camera = client.camera

                # set fov
                camera.fov = self.FoV_slider.value

                try:
                    outputs = self.get_render(self.camera_to_simple_camera(camera))
                    out = outputs["render"].cpu().detach().moveaxis(0, -1).numpy().astype(np.float32)  # chw -> hwc
                except RuntimeError as e:
                    print(e)
                    continue

                client.set_background_image(out, format="jpeg")
