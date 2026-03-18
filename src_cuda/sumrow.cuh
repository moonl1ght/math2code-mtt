#ifndef SUMROWS_CUH
#define SUMROWS_CUH

void launchSumRowsInplace(
    const float* __restrict__ A, float* __restrict__ B, int size, int batch_size
);

#endif