#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t SAMPLER = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define GLOBAL_SIZE_1_DIMS __private const int global_size_dim0,   

#define GLOBAL_SIZE_2_DIMS __private const int global_size_dim0, __private const int global_size_dim1,

#define GLOBAL_SIZE_3_DIMS \
    __private const int global_size_dim0, __private const int global_size_dim1, __private const int global_size_dim2,

#define DEAL_NON_UNIFORM_DIM1(input1)       \
    if (input1 >= global_size_dim0) {       \
        return;                             \
    }

#define DEAL_NON_UNIFORM_DIM2(input1, input2)                       \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1) { \
        return;                                                     \
    }

#define DEAL_NON_UNIFORM_DIM3(input1, input2, input3)                                             \
    if (input1 >= global_size_dim0 || input2 >= global_size_dim1 || input3 >= global_size_dim2) { \
        return;                                                                                   \
    }

#define CALCULATE_OUTPUT(i)                  \
    out##i = mad(in##i.x, weights0, out##i); \
    out##i = mad(in##i.y, weights1, out##i); \
    out##i = mad(in##i.z, weights2, out##i); \
    out##i = mad(in##i.w, weights3, out##i);

#define CALCULATE_OUTPUT_2(i)                                \
    out0_##i = mad(in0_##i.x, weights0_0, out0_##i);    \
    out0_##i = mad(in0_##i.y, weights0_1, out0_##i);    \
    out0_##i = mad(in0_##i.z, weights0_2, out0_##i);    \
    out0_##i = mad(in0_##i.w, weights0_3, out0_##i);    \
    out1_##i = mad(in1_##i.x, weights1_0, out1_##i);    \
    out1_##i = mad(in1_##i.y, weights1_1, out1_##i);    \
    out1_##i = mad(in1_##i.z, weights1_2, out1_##i);    \
    out1_##i = mad(in1_##i.w, weights1_3, out1_##i);

#define CALCULATE_OUTPUT_DOT(i)       \
    out##i.x += dot(in##i, weights0); \
    out##i.y += dot(in##i, weights1); \
    out##i.z += dot(in##i, weights2); \
    out##i.w += dot(in##i, weights3);
