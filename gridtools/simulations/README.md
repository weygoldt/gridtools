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
