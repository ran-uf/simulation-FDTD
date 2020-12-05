import numpy as np
import os
import matplotlib.pyplot as plt


def load_npy(file_dir):
    files = os.listdir(file_dir)
    data = None
    for f in files:
        if "output.npy" in f:
            if data is None:
                file_name = file_dir + '/' + f
                data = np.load(file_name)
                data = data[np.newaxis, :]
            else:
                file_name = file_dir + '/' + f
                data = np.concatenate((data, np.load(file_name)[np.newaxis, :]), axis=0)
    return data


def diffusion_calculator(sample, no_sample, ref):
    s = sample - no_sample
    r = ref - no_sample
    s_fft = np.zeros((s.shape[0], s.shape[1], s.shape[2] // 2 + 1))
    r_fft = np.zeros((r.shape[0], r.shape[1], r.shape[2] // 2 + 1))
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s_fft[i][j] = (np.abs(np.fft.rfft(s[i][j])))
            r_fft[i][j] = (np.abs(np.fft.rfft(r[i][j])))
    OCTAVE_STARTS = [8, 10, 13, 17, 22, 27, 34, 44, 55, 70, 88, 111, 140, 177, 223, 281, 354, 446]
    OCTAVE_ENDS = [11, 14, 18, 23, 28, 35, 45, 56, 71, 89, 112, 141, 178, 224, 282, 355, 447, 562]
    OCTAVE_SIZE = len(OCTAVE_STARTS)
    coefficient_s = np.zeros((s_fft.shape[0], s_fft.shape[1], OCTAVE_SIZE))
    coefficient_r = np.zeros((r_fft.shape[0], r_fft.shape[1], OCTAVE_SIZE))
    for i in range(s_fft.shape[0]):
        for j in range(s_fft.shape[1]):
            for k in range(OCTAVE_SIZE):
                s_octave = s_fft[i][j][OCTAVE_STARTS[k]:OCTAVE_ENDS[k]]
                coefficient_s[i][j][k] = sum(s_octave**2) / len(s_octave)
                r_octave = r_fft[i][j][OCTAVE_STARTS[k]:OCTAVE_ENDS[k]]
                coefficient_r[i][j][k] = sum(r_octave**2) / len(r_octave)
    coefficient_s_2 = coefficient_s ** 2
    coefficient_s_2 = np.sum(coefficient_s_2, axis=1)
    coefficient_r_2 = coefficient_r ** 2
    coefficient_r_2 = np.sum(coefficient_r_2, axis=1)

    coefficient_s_sum = np.sum(coefficient_s, axis=1)
    coefficient_r_sum = np.sum(coefficient_r, axis=1)
    coefficients_s = (coefficient_s_sum ** 2 - coefficient_s_2) / ((s.shape[1] - 1) * coefficient_s_2)
    coefficients_r = (coefficient_r_sum ** 2 - coefficient_r_2) / ((r.shape[1] - 1) * coefficient_r_2)
    coefficients = (coefficients_s - coefficients_r) / (1 - coefficients_r)
    coefficient = np.mean(coefficients, axis=0)
    return coefficient


if __name__ == "__main__":
    sample = load_npy('result/40_40_30/sin-1')
    no_sample = load_npy('result/40_40_30/no_sample')
    ref = load_npy('result/40_40_30/ref')
    coefficient_1 = diffusion_calculator(sample, no_sample, ref)
    coefficient_2 = diffusion_calculator(sample[:, :, 0:5623], no_sample[:, :, 0:5623], ref[:, :, 0:5623])
    print(coefficient_1 - coefficient_2)
    plt.plot(coefficient_1)
    plt.plot(coefficient_2)
    plt.show()
    '''
    filename_fmt = "40_20_25_%s_%s_deltaLowpass.txt"
    GEO_TYPE = ["3cylindars", "nosample", "3refcylindars"]
    SOURCE_TYPE = ["Plus30", "Plus60", "90", "Minus60", "Minus30"]
    sample = []
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[0])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[1])).T)
    sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[0], SOURCE_TYPE[2])).T)
    sample = np.array(sample)

    ref = []
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[0])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[1])).T)
    ref.append(np.loadtxt(filename_fmt % (GEO_TYPE[2], SOURCE_TYPE[2])).T)
    ref = np.array(ref)

    no_sample = []
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[0])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[1])).T)
    no_sample.append(np.loadtxt(filename_fmt % (GEO_TYPE[1], SOURCE_TYPE[2])).T)
    no_sample = np.array(no_sample)

    coefficient = diffusion_calculator(sample, no_sample, ref)
    plt.plot(coefficient)
    plt.show()
    '''


