#ifndef HIST_CUH
#define HIST_CUH

void launchNaiveHist(
    int* bins,
    const int numBins,
    const double bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int input_size,
    int threadsPerBlock
);

void launchHist(
    int* bins,
    const int numBins,
    const int bin_size,
    const int min_value,
    const int max_value,
    const int *__restrict__ input,
    const int input_size,
    int threadsPerBlock
);

#endif