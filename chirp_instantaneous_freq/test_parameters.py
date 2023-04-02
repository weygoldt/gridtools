import numpy as np 
import matplotlib.pyplot as plt
from fish_signal import chirps, wavefish_eods
from filters import bandpass_filter, instantaneous_frequency, inst_freq
from IPython import embed


def switch_test(test, defaultparams, testparams):
    if test == 'width':
        defaultparams['chirp_width'] = testparams['chirp_width']
        key = 'chirp_width'
    elif test == 'size':
        defaultparams['chirp_size'] = testparams['chirp_size']
        key = 'chirp_size'
    elif test == 'kurtosis':
        defaultparams['chirp_kurtosis'] = testparams['chirp_kurtosis']
        key = 'chirp_kurtosis'
    elif test == 'contrast':
        defaultparams['chirp_contrast'] = testparams['chirp_contrast']
        key = 'chirp_contrast'
    else:
        raise ValueError("Test not recognized")

    return key, defaultparams


def extract_dict(dict, index):
    return {key: value[index] for key, value in dict.items()}


def main(test1, test2, resolution=10):

    assert test1 in ['width', 'size', 'kurtosis', 'contrast'], "Test1 not recognized"
    assert test2 in ['width', 'size', 'kurtosis', 'contrast'], "Test2 not recognized"

    # Define the parameters for the chirp simulations 
    ntest = resolution

    defaultparams = dict(
        chirp_size = np.ones(ntest) * 100, 
        chirp_width = np.ones(ntest) * 0.1, 
        chirp_kurtosis = np.ones(ntest) * 1.0, 
        chirp_contrast = np.ones(ntest) * 0.5, 
    )

    testparams = dict(
        chirp_width = np.linspace(0.01, 0.2, ntest), 
        chirp_size = np.linspace(50, 300, ntest), 
        chirp_kurtosis = np.linspace(0.5, 1.5, ntest), 
        chirp_contrast = np.linspace(0.01, 1.0, ntest), 
    )

    key1, chirp_params = switch_test(test1, defaultparams, testparams)
    key2, chirp_params = switch_test(test2, chirp_params, testparams)

    # make the chirp trace 
    eodf = 500
    samplerate = 20000
    duration = 2
    chirp_times = [0.5, 1, 1.5]

    wide_cutoffs = 200
    tight_cutoffs = 10

    distances = np.full((ntest, ntest), np.nan)
    

    # fig, axs = plt.subplots(ntest, ntest, figsize = (10, 10), sharex = True, sharey = True)
    # axs = axs.flatten()

    iter0 = 0
    for iter1, test1_param in enumerate(chirp_params[key1]):
        for iter2, test2_param in enumerate(chirp_params[key2]):

            # get the chirp parameters for the current test
            inner_chirp_params = extract_dict(chirp_params, iter2)
            inner_chirp_params[key1] = test1_param
            inner_chirp_params[key2] = test2_param

            # make the chirp trace for the current chirp parameters
            sizes = np.ones(len(chirp_times)) * inner_chirp_params['chirp_size']
            widths = np.ones(len(chirp_times)) * inner_chirp_params['chirp_width']
            kurtosis = np.ones(len(chirp_times)) * inner_chirp_params['chirp_kurtosis']
            contrast = np.ones(len(chirp_times)) * inner_chirp_params['chirp_contrast']

            # make the chirp trace
            chirp_trace, ampmod = chirps(eodf, samplerate, duration, chirp_times, sizes, widths, kurtosis, contrast)
            signal = wavefish_eods(
                    fish="Alepto", 
                    frequency=chirp_trace, 
                    samplerate=samplerate, 
                    duration=duration, 
                    phase0=0.0, 
                    noise_std=0.05
            )           
            signal = signal * ampmod

            # apply broadband filter 
            wide_signal = bandpass_filter(signal, samplerate, eodf - wide_cutoffs, eodf + wide_cutoffs)
            tight_signal = bandpass_filter(signal, samplerate, eodf - tight_cutoffs, eodf + tight_cutoffs)

            # get the instantaneous frequency
            wide_frequency = inst_freq(wide_signal, samplerate)
            tight_frequency = inst_freq(tight_signal, samplerate)

            # bool_mask = wide_frequency != 0
            # axs[iter0].plot(wide_frequency[bool_mask])
            # axs[iter0].plot(tight_frequency[bool_mask])
            # fig.supylabel(key1)
            # fig.supxlabel(key2)

            # get the distance between the two signals
            distances[iter1, iter2] = np.linalg.norm(wide_frequency - tight_frequency)

            iter0 += 1

    fig, ax = plt.subplots()
    ax.imshow(distances, cmap = 'jet')
    plt.show()

if __name__ == "__main__":
    main('width', 'size') 
