import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d.axes3d as p3
import random

def generate_ball_trajectory(initial_pos, # m
                             bat_force, # vector, N
                             baseball_speed=40.0, # m/s
                             m_baseball=0.142, # kg
                             impulse_time=0.7e-3, # ms
                             step=0.01, # s
):
    # assuming y is parallel to ground and x is perpendicular
    v_baseball_0 = np.array((baseball_speed * np.sqrt(2), baseball_speed * np.sqrt(2), 0))
    p_baseball_0 = v_baseball_0 * m_baseball # baseball momentum
    impulse = bat_force * impulse_time # change in momentum of ball
    p_baseball_1 = p_baseball_0 + impulse # TODO: check sign
    v_baseball_1 = p_baseball_1 / m_baseball

    # given new velocity of the ball, calculate trajectory based on time steps
    # determine when the ball hits the ground (y <= 0)
    g = np.array((0, 0, -9.81)) # m/s^2
    pos = initial_pos.copy()
    t = 0.0
    Y = [pos]
    while pos[2] > 0:
        v_baseball_1 += g * step
        pos += v_baseball_1 * step
        t += step
        Y.append(pos.copy())
        
    return np.array(Y)


def animate_matplotlib(positions, # m
                       space_size, # pixels
                       time_step=0.01 # s
                       ):
    # animation funcions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ln, = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

    def init():
        x, y, z = positions[0]
        ax.plot(x, y, z)
        return ln,

    def update(n):
        x, y, z = positions[int(n)]
        ax.plot(x, y, z)
        return ln,

    # Set axis properties
    ax.set_xlim3d([0.0, space_size[0]])
    ax.set_xlabel('X')

    ax.set_ylim3d([0.0, space_size[1]])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, space_size[2]])
    ax.set_zlabel('Z')

    plt.show()

    # N = len(positions)
    # frames = np.arange(0.0, N)
    # ani = FuncAnimation(fig, update, frames=np.arange(0.0, N, 1.0), init_func=init, interval=time_step, blit=True)
    # plt.show()
    #plt.close(fig)
    #HTML(ani.to_jshtml())

def get_trajectory(bat_poses):
    mag = random.randint(int(1e3), int(6e3))
    x = random.randint(0, 30) / 100.0
    y = random.randint(0, 30) / 100.0
    z = 1.0 - x - y
    bat_force = mag * np.array((x, y, z))
    traj = generate_ball_trajectory(
        initial_pos=np.array((0.0, 0.0, 1.0)),
        bat_force=bat_force
    )
    animate_matplotlib(
        traj, np.array((130.0, 130.0, 80.0))
    )

# test
def test_trajectories():
    get_trajectory()
    # traj = generate_ball_trajectory(
    #     initial_pos=np.array((0.0, 0.0, 1.0)),
    #     bat_force=26e3 * np.array((0.4, 0.4, 0.2))
    # )

    # # test by projecting into xy and yz
    # X = traj[:, 0]
    # Y = traj[:, 1]
    # Z = traj[:, 2]

    # plt.plot(X, Y)
    # plt.show()
    # plt.plot(Y, Z)
    # plt.show()

    # print(traj)

    # animate_matplotlib(
    #     traj, np.array((300.0, 300.0, 300.0))
    # )

if __name__ == "__main__":
    test_trajectories()
