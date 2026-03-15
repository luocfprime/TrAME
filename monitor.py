from threestudio.utils.visergui import ViserViewer

if __name__ == "__main__":
    viewer = ViserViewer(viewer_port=8765)
    viewer.set_renderer(renderer_ip="localhost", renderer_port=9077)
    while True:
        viewer.update()
