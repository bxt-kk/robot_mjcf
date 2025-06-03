#! /usr/bin/env python3
import mujoco
import mujoco.viewer


def display(path:str, attach:str='', inertia:bool=False, contact_force:bool=False):
    parent:mujoco.MjSpec = mujoco.MjSpec.from_file(path)

    attaches = [s.strip() for s in attach.split(',')]
    attaches = [s for s in attaches if s]
    for i, pattern in enumerate(attaches):
        body_path, body_name, site_name = pattern.split(':')
        child:mujoco.MjSpec = mujoco.MjSpec.from_file(body_path)
        efg_main = child.find_body(body_name)
        site:mujoco.MjsSite = parent.find_site(site_name)

        site.attach_body(efg_main, prefix=f'_EXT4ATC{i:>02}_')

    m = parent.compile()
    d = mujoco.MjData(m)

    m.vis.scale.contactwidth = 0.1
    m.vis.scale.contactheight = 0.03
    m.vis.scale.forcewidth = 0.05
    m.vis.map.force = 0.05

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = inertia
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force

        viewer.sync()
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()


if __name__ == "__main__":
    import fire
    fire.Fire(display)
