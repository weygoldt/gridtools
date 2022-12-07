# GridTools

... a suite to make grid-recording analyses of weakly electric fish painfree.

This **new** version of my older [gridtools](https://github.com/weygoldt/gridtools) implementation is under development and not functional yet!
The goal is to package all preprocessing steps from the raw file to an triangulated, interpolated and annotated dataset into a single suite, including some relevant analysis functions.
Preprocessing is planned in three main steps: 

- Frequency tracking (currently: [wavetracker](https://github.com/tillraab/wavetracker))
- Triangulation & interpolation (currently: [gridtools](https://github.com/weygoldt/gridtools))
- Annotation (detection & clustering of spatio-temporal behaviors and electrocommunication)

## To do

- [ ] Refactor stoptime property of GridCleaner
- [ ] Create analysis module
- [ ] Check position estimation, seems "gritty"
- [ ] Create mock dataset to test GridCleaner and prepro
- [ ] Document GridCleaner methods in a notebook
- [ ] Implement export to .nix instead of .npy files