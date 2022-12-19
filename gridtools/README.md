# Gridtools 

A toolbox for preprocssing and position estimation of weakly electric fish on an electrode grid based on the output of the [wavetracker](https://github.com/tillraab/wavetracker)).

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Roadmap

This **new** version of my older [gridtools](https://github.com/weygoldt/gridtools) implementation is under development and not functional yet!
The goal is to package all preprocessing steps from the raw file to an triangulated, interpolated and annotated dataset into a single suite, including some relevant analysis functions.
Preprocessing is planned in three main steps: 

* Frequency tracking (currently: [wavetracker](https://github.com/tillraab/wavetracker))
* Triangulation & interpolation (currently: [gridtools](https://github.com/weygoldt/gridtools))
* Annotation (detection & clustering of spatio-temporal behaviors and electrocommunication), currently implemented in many scripts distributed on many computers, or in some cases not at all! 

- [ ] Refactor stoptime property of GridCleaner
- [ ] Create analysis module
- [ ] Implement individual Q10 estimation and norming based on that
- [ ] Check position estimation, seems "gritty"
  * Idea for better position estimates: Track the dataset well
  * Iterate through tracks and bandpass filter each track around fish frequency
  * Compute wavelet spectrogram of filtered track with higher time resolution
  * Detect chirps and extract electrode powers, maybe just rerun wavetracker on single track, also gets better track resolution.
  * But is extremely computationally expensive
- [ ] Create mock dataset to test GridCleaner and prepro
- [ ] Document GridCleaner methods in a notebook
- [ ] Implement export to .nix instead of .npy files
- [ ] Run a clustering algorith on a perfectly tracked dataset to get clusters of communication signals
- [ ] Build a perfectly labeled dataset to train a neural network to detect signals on spectrogram snippets
- [ ] Rebuild Q10 estimates to compute by binning to make it more robust


## Open questions

* What does file.flush() on nix files do?
* How do I store a np.memmap into a nix file (i.e. spectrogram)

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

## Version History

* 0.2
    * Various bug fixes and optimizations
* See [commit change]() or See [release history]()
    * 0.1
    * Initial Release

## License

    This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

    Inspiration, code snippets, etc.
