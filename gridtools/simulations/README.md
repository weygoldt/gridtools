# Simulating electrode grid recordings

This module includes tools to simulate:

- Electrode grid configurations
- Fish movement patterns based on speed and velocity
- Communication signals such as chirps and rises
- Whole grid recordigs by brining everything above together for multiple fish

# Synthetic data generation

Currently, the EOD of fish is modeled by the [thunderfish](https://github.com/janscience/thunderfish)
package. Chirps are modeled by gaussians, (a monophasic and biphasic model 
is included). Rises are modeled by a double exponential for the rise and 
decay back to baseline. The position estimates are modeled by drawing the 
heading direction and velocity for the next point in time from probability
density functions of both, which will be inferred from real data in the future.
This works but does not encompass the complex nature of real fish behavior.

## Future plans:

Modeling the EOD including communication works well for now. Movement is 
tricky, especially if it will be used for optimizing learning algorithms to
improve position estimation from electrodes alone. So heres a plan:

Build GANs (general adversarial networks). This could be able to mimic fish
movement by strining together 2 NNs, one for discrimination and one for 
generation:

- Build a discriminator network and train to seperate random numbers
from actual positions.
- Build a generator network that learns to transform random numbers
into realistic fish movement patters.

If this is trained well, we should be able to simulate infinitely many
realistic fish movements from random noise.

# Position estimation ideas

## The fish problem: 

I need an encapsulated class or function that just gets a few externally 
generated parameters such as the grid, origin, space boundaries, EOD params, etc.
and just makes a single fish, that has

- AM modulated EOD on each electrode dependent of distance, modeled by a dipole
- Wavetracker stuff
- Chirpedtector stuff
- Position ground truths

## A new tracking idea

Instead of the spatial weighted mean, use each square or triangle to 
create a vector pointing into the direction of the field. I am not 
sure what happens with fish inside squares because of the dipole though...
I need to test this. 
