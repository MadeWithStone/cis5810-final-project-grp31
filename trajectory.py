import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

def generate_ball_trajectory(initial_pos, # m
                             bat_force, # vector, N
                             baseball_speed=40.0, # m/s
                             m_baseball=0.142, # kg
                             impulse_time=0.7e-3, # ms
                             step=0.01 # s
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
                       space_size # pixels
                       ):
    # animation funcions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ln, = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

    # Set axis properties
    ax.set_xlim3d([0.0, space_size[0]])
    ax.set_xlabel('X')

    # for plotting, shift the trajectory y position to middle
    ax.set_ylim3d([-space_size[1] / 2, space_size[1] / 2])
    ax.set_ylabel('Y')

    ax.set_zlim3d([0.0, space_size[2]])
    ax.set_zlabel('Z')

    plt.show()

def get_trajectory(bat_force):
    traj = generate_ball_trajectory(
        initial_pos=np.array((0.0, 0.0, 1.0)),
        bat_force=bat_force
    )

    animate_matplotlib(
        traj, np.array((130.0, 130.0, 15.0))
    )

# test
def test_trajectories():
    get_trajectory()

if __name__ == "__main__":
    test_trajectories()
