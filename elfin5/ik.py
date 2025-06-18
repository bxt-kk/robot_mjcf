import math

from ikpy.chain import Chain
from ikpy.link import URDFLink
from numpy import ndarray
import numpy as np


class InverseKinematicsKit:

    def __init__(self):

        axis       = [0, 0, 1]
        bounds     = [-3.14, 3.14]
        bounds_235 = [-2.35, 2.35]
        bounds_261 = [-2.61, 2.61]
        bounds_256 = [-2.56, 2.56]

        self.chain = Chain(name='elfin5', links=[
            # OriginLink(),
            URDFLink(
                name='Base link',
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, -math.pi/2],
                joint_type='fixed',
            ),
            URDFLink(
                name='elfin_link1',
                origin_translation=[0, 0, 0.0735],
                origin_orientation=[0, 0, 0],
                rotation=axis,
                bounds=bounds,
            ),
            URDFLink(
                name='elfin_link2',
                origin_translation=[-0.078, 0, 0.1465],
                origin_orientation=[1.5708, 1.3734, -1.5708],
                rotation=axis,
                bounds=bounds_235,
            ),
            URDFLink(
                name='elfin_link3',
                origin_translation=[-0.37262, 0.074541, -0.0060028],
                origin_orientation=[0, 0, 1.3734],
                rotation=[0, 0, -1],
                bounds=bounds_261,
            ),
            URDFLink(
                name='elfin_link4',
                origin_translation=[0, 0.119, -0.072],
                origin_orientation=[0, 1.5708, 1.5708],
                rotation=axis,
                bounds=bounds,
            ),
            URDFLink(
                name='elfin_link5',
                origin_translation=[-0.0605, 0, 0.301],
                origin_orientation=[-1.5708, 0, 1.5708],
                rotation=[0, 0, -1],
                bounds=bounds_256,
            ),
            URDFLink(
                name='elfin_link6',
                origin_translation=[0, -0.1005, -0.0605],
                origin_orientation=[1.5708, 1.5708, 0],
                rotation=axis,
                bounds=bounds,
            ),
            URDFLink(
                name='flange_site',
                origin_translation=[0, 0, 0.054],
                origin_orientation=[0, 0, 0],
                joint_type='fixed',
            )
        ], active_links_mask=(
            [False] + [True] * 6 + [False]
        ))

    def calc_pos_params(self, qpos: ndarray):
        pos = [0.] + qpos.tolist() + [0.]
        end_m = self.chain.forward_kinematics(pos)
        end_axis = np.array([0, 0, 1]).reshape(3, 1)
        end_axis = (end_m[:3, :3] @ end_axis).flatten()
        end_pos = end_m[:3, 3]
        return (
            pos,
            end_axis,
            end_pos,
        )

    def calc_ik(
            self,
            qpos:               ndarray,
            target_position:    ndarray,
            target_orientation: ndarray,
            optimizer:          str='scalar'
        ):

        pos, _, _, = self.calc_pos_params(qpos)

        ik = self.chain.inverse_kinematics(
            target_position,
            target_orientation=target_orientation,
            orientation_mode='Z',
            initial_position=pos,
            optimizer=optimizer,
        )
        return ik


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ikpy.utils import plot
    ikk = InverseKinematicsKit()
    fig, ax = plot.init_3d_figure();
    ikk.chain.plot([0, 0, 0.47, -1.5708, 0, -1.23, 0, 0], ax)
    ax.legend()
    plt.show()
