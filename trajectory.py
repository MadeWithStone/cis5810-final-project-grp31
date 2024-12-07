import numpy as np

def generate_ball_trajectory(initial_pos, # m
                             bat_force, # vector, N
                             baseball_speed=40.0, # m/s
                             m_baseball=0.142, # kg
                             impulse_time=0.7e-3, # ms
                             step=0.01, # s
):
    # assuming y is parallel to ground and x is perpendicular
    v_baseball_0 = np.array((-baseball_speed, 0, 0))
    p_baseball_0 = v_baseball_0 * m_baseball # baseball momentum
    impulse = bat_force * impulse_time # change in momentum of ball
    p_baseball_1 = p_baseball_0 + impulse # TODO: check sign
    v_baseball_1 = p_baseball_1 / m_baseball

    # given new velocity of the ball, calculate trajectory based on time steps
    # determine when the ball hits the ground (y <= 0)
    g = np.array(0, -9.81, 0) # m/s^2
    pos = initial_pos.copy()
    t = 0.0
    X = [t]
    Y = [pos]
    while pos[1] > 0:
        v_baseball_1 += g * step
        pos += v_baseball_1 * step
        t += step
        X.append(t)
        Y.append(pos.copy())
        
    return np.array((X, Y))