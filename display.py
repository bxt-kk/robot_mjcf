#! /usr/bin/env python3
from queue import Queue
from PIL import Image
import mujoco
import mujoco.viewer
import numpy as np


CAPTURE_FNAME = 'CAPTURE'


def display(
        path:          str,
        attach:        str='',
        inertia:       bool=False,
        contact_force: bool=False,
        show_camera:   bool=False,
        sky_box:       bool=False,
        camera:        int | str=-1,
        render_size:   tuple[int, int]=(640, 360),
        depthes:       tuple[float, float, float]=(9., 3., 1.),
        print_xml:     bool=False,
    ):

    parent:mujoco.MjSpec = mujoco.MjSpec.from_file(path)

    attaches = [s.strip() for s in attach.split(',')]
    attaches = [s for s in attaches if s]
    for pattern in attaches:
        body_path, body_name, site_name = pattern.split(':')
        child:mujoco.MjSpec = mujoco.MjSpec.from_file(body_path)
        efg_main = child.body(body_name)
        site:mujoco.MjsSite = parent.site(site_name)

        site.attach_body(efg_main, prefix=f'{site_name}-')

    m = parent.compile()
    d = mujoco.MjData(m)
    if print_xml:
        print(parent.to_xml())

    m.vis.scale.contactwidth = 0.1
    m.vis.scale.contactheight = 0.03
    m.vis.scale.forcewidth = 0.05
    m.vis.map.force = 0.05

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)

    def dump_rgb_capture():
        with mujoco.Renderer(m, width=render_size[0], height=render_size[1]) as renderer:
            renderer.update_scene(d, camera=camera)
            frame = renderer.render()
            img = Image.fromarray(frame)
            dump_path = f'{CAPTURE_FNAME}.png'
            print('dump capture ->', dump_path)
            img.save(dump_path)

    def dump_depth_capture():
        with mujoco.Renderer(m, width=render_size[0], height=render_size[1]) as renderer:
            renderer.enable_depth_rendering()
            renderer.update_scene(d, camera=camera)
            frame = renderer.render()
            pixel = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.float32)
            for cix, depth in enumerate(depthes):
                normal = np.clip(frame, 0, depth)
                normal = normal - normal.min()
                normal = normal / max(1e-6, normal.max())
                pixel[..., cix] = normal
            pixel = ((1- pixel) * 255).astype(np.uint8)
            img = Image.fromarray(pixel)
            dump_path = f'{CAPTURE_FNAME}_depth.png'
            print('dump capture ->', dump_path)
            img.save(dump_path)

    event_channel = Queue(maxsize=14)

    def key_callback(keycode:int):
        keychar = chr(keycode)
        if keychar == 'C':
            event_channel.put(dump_rgb_capture)
        elif keychar == 'D':
            event_channel.put(dump_depth_capture)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = sky_box
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = inertia
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = show_camera

        viewer.sync()
        while viewer.is_running():
            while not event_channel.empty():
                event_channel.get()()
            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == "__main__":
    import fire
    fire.Fire(display)
