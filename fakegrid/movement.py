import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from scipy.stats import norm


def make_positions(origin, boundaries, trajectories, steps):
    x = np.full(len(trajectories) + 1, np.nan)
    y = np.full(len(trajectories) + 1, np.nan)
    x[0] = origin[0]
    y[0] = origin[1]

    for i in range(len(trajectories)):

        # use the fist trajectory as is
        if i == 0:
            converted_trajectory = trajectories[i]

        # make all othes trajectories relative to the previous one
        else:
            converted_trajectory = trajectories[i-1] - trajectories[i]

            # make sure the trajectory is between 0 and 2pi
            if converted_trajectory > 2*np.pi:
                converted_trajectory = converted_trajectory - 2*np.pi
            if converted_trajectory < 0:
                converted_trajectory = converted_trajectory + 2*np.pi

            # write current trajectory to trajectories to correct
            # future trajectories relative to the current one
            trajectories[i] = converted_trajectory

        # use trigonometric identities to calculate the x and y positions
        y[i+1] = np.sin(converted_trajectory) * steps[i]
        x[i+1] = np.cos(converted_trajectory) * steps[i]

    # cumulatively add the steps to the positions
    x = np.cumsum(x)
    y = np.cumsum(y)
    
    # fold back the positions if they are outside the boundaries
    boundaries = np.ravel(boundaries)
    while np.any(x < boundaries[0]) or np.any(x > boundaries[1]) or np.any(y < boundaries[2]) or np.any(y > boundaries[3]):
        x[x < boundaries[0]] = boundaries[0] + (boundaries[0] - x[x < boundaries[0]])
        x[x > boundaries[1]] = boundaries[1] - (x[x > boundaries[1]] - boundaries[1])
        y[y < boundaries[2]] = boundaries[2] + (boundaries[2] - y[y < boundaries[2]])
        y[y > boundaries[3]] = boundaries[3] - (y[y > boundaries[3]] - boundaries[3])

    return x, y


if __name__ == "__main__":

    np.random.seed(0)

    fs = 30  # sampling frequency in Hz
    tmax = 10000 # duration in seconds
    time = np.arange(0, tmax, 1/fs) # time vector
    peak_veloc = 0.2 # most common velocity in m/s
    directions = np.arange(0, 2*np.pi, 0.001) # directions vector in radians
    origin = [0, 0] # starting point
    boundaries = [(0, 10), (0, 10)] # boundaries of the arena

    # make probablitiy distribution of directions
    sigma = 7/fs
    p1 = norm.pdf(directions, 0, sigma) 
    p2 = norm.pdf(directions, np.max(directions), sigma)
    probabilities = (p1 + p2)
    probabilities = probabilities / np.sum(probabilities)

    # make random step lengths according to a gamma distribution
    step_lengths = np.random.default_rng().gamma(peak_veloc*100, 1, tmax*fs)/100
    
    # remove outliers
    step_lengths[step_lengths > 1] = 1

    # normalize to sampling rate
    step_lengths = step_lengths / fs
    
    # draw random directions according to the probability distribution
    trajectories = np.random.choice(directions, size = (tmax*fs), p = probabilities)

    # make positions
    x, y = make_positions(origin, boundaries, trajectories, step_lengths)

    fig = plt.figure()
    ax1 = plt.subplot(211, projection='polar')
    ax2 = plt.subplot(212)
    ax1.set_theta_zero_location("N")
    ax1.plot(directions, probabilities)
    ax1.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax1.set_title('Probability distribution of possible heading directions')
    ax2.hist(step_lengths * fs)
    ax2.set_title('Distribution of step lengths')
    ax2.set_xlabel('Step length (m)')
    ax2.set_ylabel('Count')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y, zorder=-10)
    ax.set_xlim(boundaries[0])
    ax.set_ylim(boundaries[1])
    ax.set_aspect('equal')
    plt.show()
