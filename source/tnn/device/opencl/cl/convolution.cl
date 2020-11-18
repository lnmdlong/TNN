#include "base.inc"
#include "activation.inc"
#include "io.inc"

__kernel void Conv2D1x1_S1_MIX(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, 
                          __global const FLOAT *weights_ptr,
                          __global const FLOAT *bias_ptr,
                          __write_only image2d_t output, __private const int2 wh,
                          __private const int input_c_blocks,
                          __private const int output_w_updiv_4) {

    const int output_cw_idx = get_global_id(0); //c/4 w/4
    const int bh_idx  = get_global_id(1); //b h

    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = vload4(output_c_block_idx, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;

    const int out_x_idx = output_w_block_idx << 2;

    int input_w_idx0 = out_x_idx;
    int input_w_idx1 = out_x_idx + 1;
    int input_w_idx2 = out_x_idx + 2;
    int input_w_idx3 = out_x_idx + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    int input_w_base   = 0;
    int weights_offset = mul24(output_c_block_idx, input_c_blocks << 2);
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        weights0 = vload4(weights_offset, (__global FLOAT *)weights_ptr);
        weights1 = vload4(weights_offset + 1, (__global FLOAT *)weights_ptr);
        weights2 = vload4(weights_offset + 2, (__global FLOAT *)weights_ptr);
        weights3 = vload4(weights_offset + 3, (__global FLOAT *)weights_ptr);

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_offset += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, wh.x);

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1_S1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4) {
    const int output_cw_idx = get_global_id(0);
    const int bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = output_w_block_idx << 2;
    int input_w_idx1 = input_w_idx0 + 1;
    int input_w_idx2 = input_w_idx0 + 2;
    int input_w_idx3 = input_w_idx0 + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_w_base += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int input_c_blocks, __private const int2 output_wh,
    __private const int2 stride_wh, __private const int output_w_updiv_4) {
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int output_c_block_idx = output_cw_idx / output_w_updiv_4;
    const int output_w_block_idx = output_cw_idx % output_w_updiv_4;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = mul24(output_w_block_idx, stride_wh.x << 2);
    int input_w_idx1 = input_w_idx0 + stride_wh.x;
    int input_w_idx2 = input_w_idx1 + stride_wh.x;
    int input_w_idx3 = input_w_idx2 + stride_wh.x;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= input_wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= input_wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= input_wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= input_wh.x);

    int b_idx = output_bh_idx / output_wh.y;
    int input_bh_idx = mad24(output_bh_idx % output_wh.y, stride_wh.y, b_idx * input_wh.y);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        int input_w_base   = input_c_block_idx * input_wh.x;
        int weights_w_base = input_c_block_idx << 2;

        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, input_bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, input_bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, input_bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, input_bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, output_wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D1x1GS3D_S1(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 wh,
    __private const int input_c_blocks,
    __private const int output_w_updiv_4) {
    const int output_c_block_idx = get_global_id(0);
    const int output_w_block_idx = get_global_id(1);
    const int bh_idx      = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(output_c_block_idx, output_w_block_idx, bh_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = output_w_block_idx << 2;
    int input_w_idx1 = input_w_idx0 + 1;
    int input_w_idx2 = input_w_idx0 + 2;
    int input_w_idx3 = input_w_idx0 + 3;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base   += wh.x;
        weights_w_base += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               bh_idx, remain);
}

__kernel void Conv2D1x1GS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input, /* [w,h] [c%4 * w * c/4, h * b] */
    __read_only image2d_t weights,                  /* [w,h] [cout%4 * cin, cout/4] */
    __read_only image2d_t bias,                     /* [w,h] [cout%4 * cout/4, 1]   */
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int input_c_blocks, __private const int2 output_wh,
    __private const int2 stride_wh, __private const int output_w_updiv_4) {
    const int output_c_block_idx = get_global_id(0);
    const int output_w_block_idx = get_global_id(1);
    const int output_bh_idx      = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(output_c_block_idx, output_w_block_idx, output_bh_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(output_c_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int input_w_idx0 = mul24(output_w_block_idx, stride_wh.x << 2);
    int input_w_idx1 = input_w_idx0 + stride_wh.x;
    int input_w_idx2 = input_w_idx1 + stride_wh.x;
    int input_w_idx3 = input_w_idx2 + stride_wh.x;

    input_w_idx0 = select(input_w_idx0, INT_MIN, input_w_idx0 >= input_wh.x);
    input_w_idx1 = select(input_w_idx1, INT_MIN, input_w_idx1 >= input_wh.x);
    input_w_idx2 = select(input_w_idx2, INT_MIN, input_w_idx2 >= input_wh.x);
    input_w_idx3 = select(input_w_idx3, INT_MIN, input_w_idx3 >= input_wh.x);

    int b_idx = output_bh_idx / output_wh.y;
    int input_bh_idx = mul24((output_bh_idx % output_wh.y), stride_wh.y) + b_idx * input_wh.y;

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    int input_w_base   = 0;
    int weights_w_base = 0;
    for (int input_c_block_idx = 0; input_c_block_idx < input_c_blocks; ++input_c_block_idx) {
        in0 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx0, input_bh_idx));
        in1 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx1, input_bh_idx));
        in2 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx2, input_bh_idx));
        in3 = RI_F(input, SAMPLER, (int2)(input_w_base + input_w_idx3, input_bh_idx));

        weights0 = RI_F(weights, SAMPLER, (int2)(weights_w_base, output_c_block_idx));
        weights1 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 1, output_c_block_idx));
        weights2 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 2, output_c_block_idx));
        weights3 = RI_F(weights, SAMPLER, (int2)(weights_w_base + 3, output_c_block_idx));

        CALCULATE_OUTPUT(0);
        CALCULATE_OUTPUT(1);
        CALCULATE_OUTPUT(2);
        CALCULATE_OUTPUT(3);

        input_w_base += input_wh.x;
        weights_w_base += 4;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(output_c_block_idx, output_wh.x);
    int out_x_idx        = output_w_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int out_channel_block_idx = output_cw_idx / out_width_blocks;
    const int out_width_block_idx   = output_cw_idx % out_width_blocks;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) + 
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);

                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2D3x3s1d1(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int output_cw_idx = get_global_id(0);
    const int output_bh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bh_idx);

    const int out_channel_block_idx = output_cw_idx / out_width_blocks;
    const int out_width_block_idx   = output_cw_idx % out_width_blocks;
    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));

    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    int weights_y_idx = weights_h_idx;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_x_idx = input_c_block_idx << 2;
            #if 1
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);
            #else
            READ_INPUT_IMAGE(0, 0);
            in1 = in0 + 1;
            in2 = in0 + 2;
            in3 = in0 + 3;
            in4 = in0 + 4;
            #endif

            #if 1
            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
            weights_w0_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
            weights_w0_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
            weights_w0_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx));

            weights_w1_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx + 1));
            weights_w1_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx + 1));
            weights_w1_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx + 1));
            weights_w1_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx + 1));

            weights_w2_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx + 2));
            weights_w2_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx + 2));
            weights_w2_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx + 2));
            weights_w2_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx + 2));
            #else
            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
            weights_w0_c1 = weights_w0_c0 + 111;
            weights_w0_c2 = weights_w0_c0 + 222;
            weights_w0_c3 = weights_w0_c0 + 333;
            weights_w1_c0 = weights_w0_c0 + 44;
            weights_w1_c1 = weights_w0_c0 + 555;
            weights_w1_c2 = weights_w0_c0 + 66;
            weights_w1_c3 = weights_w0_c0 + 777;
            weights_w2_c0 = weights_w0_c0 + 88;
            weights_w2_c1 = weights_w0_c0 + 99;
            weights_w2_c2 = weights_w0_c0 + 100;
            weights_w2_c3 = weights_w0_c0 + 1132;
            #endif

            #if 1
            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);

            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);

            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);

            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);
            #else
            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);

            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);

            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);

            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);

            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);

            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);

            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);

            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);

            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);

            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);

            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);

            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);
            #endif
        }
        weights_y_idx += 3;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2DGS3D3x3s1d1(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
#if 1
    __read_only image2d_t weights,
#else
    __read_only image2d_t weights0, __read_only image2d_t weights1,
    __read_only image2d_t weights2, __read_only image2d_t weights3,
#endif
    __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    #if 1
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);
    #elif 1
    const int out_width_block_idx   = get_global_id(0);
    const int output_bh_idx         = get_global_id(1);
    const int out_channel_block_idx = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_width_block_idx, output_bh_idx, out_channel_block_idx);
    #endif

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));

    bool is_w0_in_boundary = (in_width0 >= 0 || in_width0 < input_wh.x);
    bool is_w1_in_boundary = (in_width1 >= 0 || in_width1 < input_wh.x);
    bool is_w2_in_boundary = (in_width2 >= 0 || in_width2 < input_wh.x);
    bool is_w3_in_boundary = (in_width3 >= 0 || in_width3 < input_wh.x);
    bool is_w4_in_boundary = (in_width4 >= 0 || in_width4 < input_wh.x);
    bool is_w5_in_boundary = (in_width5 >= 0 || in_width5 < input_wh.x);
    
    FLOAT4 zero = (FLOAT4)0.0f;
    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    int3 weights_y_idx = (int3)(weights_h_idx, weights_h_idx + 1, weights_h_idx + 2);
    // int3 weights_y_idx = (int3)(weights_h_idx, weights_h_idx, weights_h_idx);

    int magic_num = padding_wh.x + 1;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        #if 1
        int in_idx = 0;
        int4 weights_x_idx = (int4)(0, 1, 2, 3);
        #else
        int4 weights_x_idx = (int4)0;
        #endif
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            #if 0
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_x_idx_c0 = mul24(input_c_block_idx, 4);
            int weights_x_idx_c1 = weights_x_idx_c0 + 1;
            int weights_x_idx_c2 = weights_x_idx_c0 + 2;
            int weights_x_idx_c3 = weights_x_idx_c0 + 3;
            int weights_y_idx_w1 = weights_y_idx + 1;
            int weights_y_idx_w2 = weights_y_idx + 2;
            #if 0
            for (int i = 0; i < 10; i++) {
                weights_x_idx_c1 *= (magic_num - 1);
            }
            #endif
            #elif 1
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            // int weights_x_idx_c0 = mul24(input_c_block_idx, 4);
            // int weights_x_idx_c1 = weights_x_idx_c0;
            // int weights_x_idx_c2 = weights_x_idx_c0;
            // int weights_x_idx_c3 = weights_x_idx_c0;
            // int weights_y_idx_w1 = weights_y_idx + 1;
            // int weights_y_idx_w2 = weights_y_idx + 2;
            #endif
            #if 0
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);
            #elif 1
            in0 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width0, is_w0_in_boundary), in_hb_value));
            in1 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width1, is_w1_in_boundary), in_hb_value));
            in2 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width2, is_w2_in_boundary), in_hb_value));
            in3 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width3, is_w3_in_boundary), in_hb_value));
            in4 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width4, is_w4_in_boundary), in_hb_value));
            in5 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width5, is_w5_in_boundary), in_hb_value));
            #elif 1
            in0 = RI_F(input, SAMPLER, (int2)(in_idx + in_width0, in_hb_value));
            in1 = RI_F(input, SAMPLER, (int2)(in_idx + in_width1, in_hb_value));
            in2 = RI_F(input, SAMPLER, (int2)(in_idx + in_width2, in_hb_value));
            in3 = RI_F(input, SAMPLER, (int2)(in_idx + in_width3, in_hb_value));
            in4 = RI_F(input, SAMPLER, (int2)(in_idx + in_width4, in_hb_value));
            in5 = RI_F(input, SAMPLER, (int2)(in_idx + in_width5, in_hb_value));
            #endif

            #if 1
            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x));
            weights_w0_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x));
            weights_w0_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x));
            weights_w0_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));

            weights_w1_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
            weights_w1_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
            weights_w1_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
            weights_w1_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));

            weights_w2_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.z));
            weights_w2_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.z));
            weights_w2_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.z));
            weights_w2_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.z));
            #elif 0
            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.x));
            weights_w0_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.x));
            weights_w0_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.x));
            weights_w0_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.x));
            weights_w1_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.y));
            weights_w1_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.y));
            weights_w1_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.y));
            weights_w1_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.y));
            weights_w2_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.x, weights_y_idx.z));
            weights_w2_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.y, weights_y_idx.z));
            weights_w2_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.z, weights_y_idx.z));
            weights_w2_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx.w, weights_y_idx.z));
            #elif 0
            weights_w0_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w1_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w2_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w0_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w1_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w2_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w0_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w1_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w2_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w0_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w1_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w2_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            #elif 1
            weights_w0_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.x));
            weights_w0_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.x));
            weights_w0_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.x));
            weights_w0_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.x));
            weights_w1_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.y));
            weights_w1_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.y));
            weights_w1_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.y));
            weights_w1_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.y));
            weights_w2_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.z));
            weights_w2_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.z));
            weights_w2_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.z));
            weights_w2_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx.z));
            #endif

#if 1
            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);
            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);
            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);
            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);
#elif 1
            out0 += in0.x * weights_w0_c0;
            out0 += in0.y * weights_w0_c1;
            out0 += in0.z * weights_w0_c2;
            out0 += in0.w * weights_w0_c3;
            out0 += in1.x * weights_w1_c0;
            out0 += in1.y * weights_w1_c1;
            out0 += in1.z * weights_w1_c2;
            out0 += in1.w * weights_w1_c3;
            out0 += in2.x * weights_w2_c0;
            out0 += in2.y * weights_w2_c1;
            out0 += in2.z * weights_w2_c2;
            out0 += in2.w * weights_w2_c3;
            out1 += in1.x * weights_w0_c0;
            out1 += in1.y * weights_w0_c1;
            out1 += in1.z * weights_w0_c2;
            out1 += in1.w * weights_w0_c3;
            out1 += in2.x * weights_w1_c0;
            out1 += in2.y * weights_w1_c1;
            out1 += in2.z * weights_w1_c2;
            out1 += in2.w * weights_w1_c3;
            out1 += in3.x * weights_w2_c0;
            out1 += in3.y * weights_w2_c1;
            out1 += in3.z * weights_w2_c2;
            out1 += in3.w * weights_w2_c3;
            out2 += in2.x * weights_w0_c0;
            out2 += in2.y * weights_w0_c1;
            out2 += in2.z * weights_w0_c2;
            out2 += in2.w * weights_w0_c3;
            out2 += in3.x * weights_w1_c0;
            out2 += in3.y * weights_w1_c1;
            out2 += in3.z * weights_w1_c2;
            out2 += in3.w * weights_w1_c3;
            out2 += in4.x * weights_w2_c0;
            out2 += in4.y * weights_w2_c1;
            out2 += in4.z * weights_w2_c2;
            out2 += in4.w * weights_w2_c3;
            out3 += in3.x * weights_w0_c0;
            out3 += in3.y * weights_w0_c1;
            out3 += in3.z * weights_w0_c2;
            out3 += in3.w * weights_w0_c3;
            out3 += in4.x * weights_w1_c0;
            out3 += in4.y * weights_w1_c1;
            out3 += in4.z * weights_w1_c2;
            out3 += in4.w * weights_w1_c3;
            out3 += in5.x * weights_w2_c0;
            out3 += in5.y * weights_w2_c1;
            out3 += in5.z * weights_w2_c2;
            out3 += in5.w * weights_w2_c3;
#elif 1
            out0 += in0;
            out0 += in1;
            out0 += in2;
            out0 += in3;
            out0 += in4;
            out0 += in5;
            out0 += weights_w0_c0;
            out0 += weights_w0_c1;
            out0 += weights_w0_c2;
            out0 += weights_w0_c3;
            out0 += weights_w1_c0;
            out0 += weights_w1_c1;
            out0 += weights_w1_c2;
            out0 += weights_w1_c3;
            out0 += weights_w2_c0;
            out0 += weights_w2_c1;
            out0 += weights_w2_c2;
            out0 += weights_w2_c3;
#endif

            #if 0
            FLOAT4 out_debug = weights_y_idx;
            if (out_channel_block_idx == 0 && out_width_block_idx == 0 &&
                output_bh_idx == 0 && input_c_block_idx == in_channel_block_length - 1) {
                WI_F(output, (int2)(0, 0), out_debug);
            }
            #endif

            #if 1
            // in_idx += input_wh.x;
            weights_x_idx += 4;
            // weights_x_idx.x++;
            #endif
        }
        weights_y_idx += 3;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 1
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
#else
    WI_F(output, (int2)(output_w_idx, output_bh_idx), out0);
    if (remain >= 4) {
        WI_F(output, (int2)(output_w_idx + 3, output_bh_idx), out3);
    }
    if (remain >= 3) {
        WI_F(output, (int2)(output_w_idx + 2, output_bh_idx), out2);
    }
    if (remain >= 2) {
        WI_F(output, (int2)(output_w_idx + 1, output_bh_idx), out1);
    }
#endif
}

__kernel void Conv2DGS3D3x3s1d1_MIX(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __global const FLOAT *weights_ptr,
    __global const FLOAT *bias_ptr,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    #if 0
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    #elif 1
    const int out_width_block_idx   = get_global_id(0);
    const int output_bh_idx         = get_global_id(1);
    const int out_channel_block_idx = get_global_id(2);
    #endif
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;
    const int input_channels = 80;

    FLOAT4 out0 = vload4(out_channel_block_idx, (__global FLOAT *)bias_ptr);
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));

    bool is_w0_in_boundary = (in_width0 >= 0 || in_width0 < input_wh.x);
    bool is_w1_in_boundary = (in_width1 >= 0 || in_width1 < input_wh.x);
    bool is_w2_in_boundary = (in_width2 >= 0 || in_width2 < input_wh.x);
    bool is_w3_in_boundary = (in_width3 >= 0 || in_width3 < input_wh.x);
    bool is_w4_in_boundary = (in_width4 >= 0 || in_width4 < input_wh.x);
    bool is_w5_in_boundary = (in_width5 >= 0 || in_width5 < input_wh.x);
    
    FLOAT4 zero = (FLOAT4)0.0f;
    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    int3 weights_y_idx = (int3)(weights_h_idx, weights_h_idx + 1, weights_h_idx + 2);
    // int3 weights_y_idx = (int3)(weights_h_idx, weights_h_idx, weights_h_idx);

    int channel_offset = input_channels;
    __global FLOAT4* weights_w0_c0_ptr = (__global FLOAT4*)weights_ptr;
    __global FLOAT4* weights_w0_c1_ptr = weights_w0_c0_ptr + 1;
    __global FLOAT4* weights_w0_c2_ptr = weights_w0_c0_ptr + 2;
    __global FLOAT4* weights_w0_c3_ptr = weights_w0_c0_ptr + 3;
    __global FLOAT4* weights_w1_c0_ptr = weights_w0_c0_ptr + channel_offset;
    __global FLOAT4* weights_w1_c1_ptr = weights_w1_c0_ptr + 1;
    __global FLOAT4* weights_w1_c2_ptr = weights_w1_c0_ptr + 2;
    __global FLOAT4* weights_w1_c3_ptr = weights_w1_c0_ptr + 3;
    __global FLOAT4* weights_w2_c0_ptr = weights_w1_c0_ptr + channel_offset;
    __global FLOAT4* weights_w2_c1_ptr = weights_w2_c0_ptr + 1;
    __global FLOAT4* weights_w2_c2_ptr = weights_w2_c0_ptr + 2;
    __global FLOAT4* weights_w2_c3_ptr = weights_w2_c0_ptr + 3;

    int magic_num = padding_wh.x + 1;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        #if 0
        int in_idx = 0;
        int4 weights_x_idx = (int4)(0, 1, 2, 3);
        #else
        int4 weights_x_idx = (int4)0;
        #endif
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            #if 0
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_x_idx_c0 = mul24(input_c_block_idx, 4);
            int weights_x_idx_c1 = weights_x_idx_c0 + 1;
            int weights_x_idx_c2 = weights_x_idx_c0 + 2;
            int weights_x_idx_c3 = weights_x_idx_c0 + 3;
            int weights_y_idx_w1 = weights_y_idx + 1;
            int weights_y_idx_w2 = weights_y_idx + 2;
            #if 0
            for (int i = 0; i < 10; i++) {
                weights_x_idx_c1 *= (magic_num - 1);
            }
            #endif
            #elif 1
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            // int weights_x_idx_c0 = mul24(input_c_block_idx, 4);
            // int weights_x_idx_c1 = weights_x_idx_c0;
            // int weights_x_idx_c2 = weights_x_idx_c0;
            // int weights_x_idx_c3 = weights_x_idx_c0;
            // int weights_y_idx_w1 = weights_y_idx + 1;
            // int weights_y_idx_w2 = weights_y_idx + 2;
            #endif
            #if 0
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);
            #elif 1
            in0 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width0, is_w0_in_boundary), in_hb_value));
            in1 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width1, is_w1_in_boundary), in_hb_value));
            in2 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width2, is_w2_in_boundary), in_hb_value));
            in3 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width3, is_w3_in_boundary), in_hb_value));
            in4 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width4, is_w4_in_boundary), in_hb_value));
            in5 = RI_F(input, SAMPLER, (int2)(select(-1, in_idx + in_width5, is_w5_in_boundary), in_hb_value));
            #elif 1
            in0 = RI_F(input, SAMPLER, (int2)(in_idx + in_width0, in_hb_value));
            in1 = RI_F(input, SAMPLER, (int2)(in_idx + in_width1, in_hb_value));
            in2 = RI_F(input, SAMPLER, (int2)(in_idx + in_width2, in_hb_value));
            in3 = RI_F(input, SAMPLER, (int2)(in_idx + in_width3, in_hb_value));
            in4 = RI_F(input, SAMPLER, (int2)(in_idx + in_width4, in_hb_value));
            in5 = RI_F(input, SAMPLER, (int2)(in_idx + in_width5, in_hb_value));
            #endif

            #if 0
            int weights_offset = mad24(weights_y_idx.x, input_channels, input_c_block_idx << 2);
            weights_w0_c0 = weights_w0_c0_ptr[weights_offset];
            weights_w0_c1 = weights_w0_c1_ptr[weights_offset];
            weights_w0_c2 = weights_w0_c2_ptr[weights_offset];
            weights_w0_c3 = weights_w0_c3_ptr[weights_offset];
            weights_w1_c0 = weights_w1_c0_ptr[weights_offset];
            weights_w1_c1 = weights_w1_c1_ptr[weights_offset];
            weights_w1_c2 = weights_w1_c2_ptr[weights_offset];
            weights_w1_c3 = weights_w1_c3_ptr[weights_offset];
            weights_w2_c0 = weights_w2_c0_ptr[weights_offset];
            weights_w2_c1 = weights_w2_c1_ptr[weights_offset];
            weights_w2_c2 = weights_w2_c2_ptr[weights_offset];
            weights_w2_c3 = weights_w2_c3_ptr[weights_offset];
            #else
            int weights_offset = mad24(weights_y_idx.x, input_channels, input_c_block_idx << 2);
            int weights_offset_w1 = weights_offset + channel_offset;
            int weights_offset_w2 = weights_offset_w1 + channel_offset;
            weights_w0_c0 = weights_w0_c0_ptr[weights_offset];
            weights_w0_c1 = weights_w0_c0_ptr[weights_offset + 1];
            weights_w0_c2 = weights_w0_c0_ptr[weights_offset + 2];
            weights_w0_c3 = weights_w0_c0_ptr[weights_offset + 3];
            weights_w1_c0 = weights_w0_c0_ptr[weights_offset_w1];
            weights_w1_c1 = weights_w0_c0_ptr[weights_offset_w1 + 1];
            weights_w1_c2 = weights_w0_c0_ptr[weights_offset_w1 + 2];
            weights_w1_c3 = weights_w0_c0_ptr[weights_offset_w1 + 3];
            weights_w2_c0 = weights_w0_c0_ptr[weights_offset_w2];
            weights_w2_c1 = weights_w0_c0_ptr[weights_offset_w2 + 1];
            weights_w2_c2 = weights_w0_c0_ptr[weights_offset_w2 + 2];
            weights_w2_c3 = weights_w0_c0_ptr[weights_offset_w2 + 3];
            #endif

#if 1
            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);
            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);
            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);
            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);
#elif 1
            out0 += in0.x * weights_w0_c0;
            out0 += in0.y * weights_w0_c1;
            out0 += in0.z * weights_w0_c2;
            out0 += in0.w * weights_w0_c3;
            out0 += in1.x * weights_w1_c0;
            out0 += in1.y * weights_w1_c1;
            out0 += in1.z * weights_w1_c2;
            out0 += in1.w * weights_w1_c3;
            out0 += in2.x * weights_w2_c0;
            out0 += in2.y * weights_w2_c1;
            out0 += in2.z * weights_w2_c2;
            out0 += in2.w * weights_w2_c3;
            out1 += in1.x * weights_w0_c0;
            out1 += in1.y * weights_w0_c1;
            out1 += in1.z * weights_w0_c2;
            out1 += in1.w * weights_w0_c3;
            out1 += in2.x * weights_w1_c0;
            out1 += in2.y * weights_w1_c1;
            out1 += in2.z * weights_w1_c2;
            out1 += in2.w * weights_w1_c3;
            out1 += in3.x * weights_w2_c0;
            out1 += in3.y * weights_w2_c1;
            out1 += in3.z * weights_w2_c2;
            out1 += in3.w * weights_w2_c3;
            out2 += in2.x * weights_w0_c0;
            out2 += in2.y * weights_w0_c1;
            out2 += in2.z * weights_w0_c2;
            out2 += in2.w * weights_w0_c3;
            out2 += in3.x * weights_w1_c0;
            out2 += in3.y * weights_w1_c1;
            out2 += in3.z * weights_w1_c2;
            out2 += in3.w * weights_w1_c3;
            out2 += in4.x * weights_w2_c0;
            out2 += in4.y * weights_w2_c1;
            out2 += in4.z * weights_w2_c2;
            out2 += in4.w * weights_w2_c3;
            out3 += in3.x * weights_w0_c0;
            out3 += in3.y * weights_w0_c1;
            out3 += in3.z * weights_w0_c2;
            out3 += in3.w * weights_w0_c3;
            out3 += in4.x * weights_w1_c0;
            out3 += in4.y * weights_w1_c1;
            out3 += in4.z * weights_w1_c2;
            out3 += in4.w * weights_w1_c3;
            out3 += in5.x * weights_w2_c0;
            out3 += in5.y * weights_w2_c1;
            out3 += in5.z * weights_w2_c2;
            out3 += in5.w * weights_w2_c3;
#elif 1
            out0 += in0;
            out0 += in1;
            out0 += in2;
            out0 += in3;
            out0 += in4;
            out0 += in5;
            out0 += weights_w0_c0;
            out0 += weights_w0_c1;
            out0 += weights_w0_c2;
            out0 += weights_w0_c3;
            out0 += weights_w1_c0;
            out0 += weights_w1_c1;
            out0 += weights_w1_c2;
            out0 += weights_w1_c3;
            out0 += weights_w2_c0;
            out0 += weights_w2_c1;
            out0 += weights_w2_c2;
            out0 += weights_w2_c3;
#endif

            #if 0
            FLOAT4 out_debug = weights_y_idx;
            if (out_channel_block_idx == 0 && out_width_block_idx == 0 &&
                output_bh_idx == 0 && input_c_block_idx == in_channel_block_length - 1) {
                WI_F(output, (int2)(0, 0), out_debug);
            }
            #endif

            #if 0
            // in_idx += input_wh.x;
            weights_x_idx += 4;
            // weights_x_idx.x++;
            #endif
        }
        weights_y_idx += 3;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 1
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
#else
    
#endif
}

__kernel void Conv2DGS3D3x3s1d1Multi(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights0, __read_only image2d_t weights1,
    __read_only image2d_t weights2, __read_only image2d_t weights3,
    __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));

    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    int weights_y_idx = weights_h_idx;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        #if 0
        int in_idx = 0;
        int weights_x_idx = 0;
        #endif
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            #if 1
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_y_idx_w1 = weights_y_idx + 1;
            int weights_y_idx_w2 = weights_y_idx + 2;
            #endif
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);

            weights_w0_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w0_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w0_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
            weights_w0_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));

            weights_w1_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w1_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w1_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));
            weights_w1_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w1));

            weights_w2_c0 = RI_F(weights0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w2_c1 = RI_F(weights1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w2_c2 = RI_F(weights2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));
            weights_w2_c3 = RI_F(weights3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx_w2));

#if 1
            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);
#else
            out0 += in0.x + weights_w0_c0;
            out0 += in0.y + weights_w0_c1;
            out0 += in0.z + weights_w0_c2;
            out0 += in0.w + weights_w0_c3;
            out0 += in1.x + weights_w1_c0;
            out0 += in1.y + weights_w1_c1;
            out0 += in1.z + weights_w1_c2;
            out0 += in1.w + weights_w1_c3;
            out0 += in2.x + weights_w2_c0;
            out0 += in2.y + weights_w2_c1;
            out0 += in2.z + weights_w2_c2;
            out0 += in2.w + weights_w2_c3;
#endif

            #if 0
            FLOAT4 out_debug = weights_y_idx;
            if (out_channel_block_idx == 0 && out_width_block_idx == 0 &&
                output_bh_idx == 0 && input_c_block_idx == in_channel_block_length - 1) {
                WI_F(output, (int2)(0, 0), out_debug);
            }
            #endif

#if 1
            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);
#endif

            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);

            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);

            #if 0
            in_idx += input_wh.x;
            weights_x_idx += 4;
            #endif
        }
        weights_y_idx += 3;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 1
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
#else
    
#endif
}

__kernel void Conv2DGS3D3x3s1d1LoopRearrange(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));

    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy++) {
            int in_hb_value = iy + batch_idx;
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);

            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
            weights_w0_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
            weights_w0_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
            weights_w0_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx));

            weights_w1_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx + 1));
            weights_w1_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx + 1));
            weights_w1_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx + 1));
            weights_w1_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx + 1));

            weights_w2_c0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx + 2));
            weights_w2_c1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx + 2));
            weights_w2_c2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx + 2));
            weights_w2_c3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx + 2));

            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);

            #if 0
            FLOAT4 out_debug = weights_y_idx;
            if (out_channel_block_idx == 0 && out_width_block_idx == 0 &&
                output_bh_idx == 0 && input_c_block_idx == in_channel_block_length - 1) {
                WI_F(output, (int2)(0, 0), out_debug);
            }
            #endif

            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);

            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);

            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);

            weights_y_idx += 3;
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 1
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
#else
    
#endif
}

__kernel void Conv2DGS3D3x3s1d1Rearrange(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks, __private const int input_channel) {
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx = mul24(in_height_start - height_start, mul24(input_channel, 3));

    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0, weights_w0_c1, weights_w0_c2, weights_w0_c3;
    FLOAT4 weights_w1_c0, weights_w1_c1, weights_w1_c2, weights_w1_c3;
    FLOAT4 weights_w2_c0, weights_w2_c1, weights_w2_c2, weights_w2_c3;
    int weights_y_idx_w0 = weights_h_idx;
    int weights_y_idx_w1 = weights_h_idx + input_channel;
    int weights_y_idx_w2 = weights_y_idx_w1 + input_channel;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_x_idx = input_c_block_idx << 2;
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);

            weights_w0_c0 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w0 + weights_x_idx));
            weights_w0_c1 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w0 + weights_x_idx + 1));
            weights_w0_c2 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w0 + weights_x_idx + 2));
            weights_w0_c3 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w0 + weights_x_idx + 3));

            weights_w1_c0 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w1 + weights_x_idx));
            weights_w1_c1 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w1 + weights_x_idx + 1));
            weights_w1_c2 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w1 + weights_x_idx + 2));
            weights_w1_c3 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w1 + weights_x_idx + 3));

            weights_w2_c0 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w2 + weights_x_idx));
            weights_w2_c1 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w2 + weights_x_idx + 1));
            weights_w2_c2 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w2 + weights_x_idx + 2));
            weights_w2_c3 = RI_F(weights, SAMPLER, (int2)(out_channel_block_idx, weights_y_idx_w2 + weights_x_idx + 3));

            out0 = mad(in0.x, weights_w0_c0, out0);
            out0 = mad(in0.y, weights_w0_c1, out0);
            out0 = mad(in0.z, weights_w0_c2, out0);
            out0 = mad(in0.w, weights_w0_c3, out0);
            out0 = mad(in1.x, weights_w1_c0, out0);
            out0 = mad(in1.y, weights_w1_c1, out0);
            out0 = mad(in1.z, weights_w1_c2, out0);
            out0 = mad(in1.w, weights_w1_c3, out0);
            out0 = mad(in2.x, weights_w2_c0, out0);
            out0 = mad(in2.y, weights_w2_c1, out0);
            out0 = mad(in2.z, weights_w2_c2, out0);
            out0 = mad(in2.w, weights_w2_c3, out0);

            #if 0
            FLOAT4 out_debug = weights_y_idx_w0 + 1;
            if (out_channel_block_idx == 0 && out_width_block_idx == 0 &&
                output_bh_idx == 0 && input_c_block_idx == in_channel_block_length - 1) {
                WI_F(output, (int2)(0, 0), out_debug);
            }
            #endif

            out1 = mad(in1.x, weights_w0_c0, out1);
            out1 = mad(in1.y, weights_w0_c1, out1);
            out1 = mad(in1.z, weights_w0_c2, out1);
            out1 = mad(in1.w, weights_w0_c3, out1);
            out1 = mad(in2.x, weights_w1_c0, out1);
            out1 = mad(in2.y, weights_w1_c1, out1);
            out1 = mad(in2.z, weights_w1_c2, out1);
            out1 = mad(in2.w, weights_w1_c3, out1);
            out1 = mad(in3.x, weights_w2_c0, out1);
            out1 = mad(in3.y, weights_w2_c1, out1);
            out1 = mad(in3.z, weights_w2_c2, out1);
            out1 = mad(in3.w, weights_w2_c3, out1);

            out2 = mad(in2.x, weights_w0_c0, out2);
            out2 = mad(in2.y, weights_w0_c1, out2);
            out2 = mad(in2.z, weights_w0_c2, out2);
            out2 = mad(in2.w, weights_w0_c3, out2);
            out2 = mad(in3.x, weights_w1_c0, out2);
            out2 = mad(in3.y, weights_w1_c1, out2);
            out2 = mad(in3.z, weights_w1_c2, out2);
            out2 = mad(in3.w, weights_w1_c3, out2);
            out2 = mad(in4.x, weights_w2_c0, out2);
            out2 = mad(in4.y, weights_w2_c1, out2);
            out2 = mad(in4.z, weights_w2_c2, out2);
            out2 = mad(in4.w, weights_w2_c3, out2);

            out3 = mad(in3.x, weights_w0_c0, out3);
            out3 = mad(in3.y, weights_w0_c1, out3);
            out3 = mad(in3.z, weights_w0_c2, out3);
            out3 = mad(in3.w, weights_w0_c3, out3);
            out3 = mad(in4.x, weights_w1_c0, out3);
            out3 = mad(in4.y, weights_w1_c1, out3);
            out3 = mad(in4.z, weights_w1_c2, out3);
            out3 = mad(in4.w, weights_w1_c3, out3);
            out3 = mad(in5.x, weights_w2_c0, out3);
            out3 = mad(in5.y, weights_w2_c1, out3);
            out3 = mad(in5.z, weights_w2_c2, out3);
            out3 = mad(in5.w, weights_w2_c3, out3);
        }
        weights_y_idx_w0 += input_channel * 3;
        weights_y_idx_w1 += input_channel * 3;
        weights_y_idx_w2 += input_channel * 3;
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 1
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
#endif
}

__kernel void Conv2DGS3D3x3s1d1cb2(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int out_channel_block_length,
    __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    const int out_channel_slice_idx = get_global_id(0);
    const int out_channel_block_idx = out_channel_slice_idx << 1;
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_slice_idx, out_width_block_idx, output_bh_idx);

    const int out_batch_idx  = output_bh_idx / output_wh.y;
    const int out_height_idx = output_bh_idx % output_wh.y;
    bool is_s1_in_boundary = (out_channel_block_idx + 1 < out_channel_block_length);

    FLOAT4 out_w0_s0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;
    FLOAT4 out_w0_s1 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx + 1, 0));
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;


    int in_width0 = (out_width_block_idx << 2) - padding_wh.x;
    int in_width1 = in_width0 + 1;
    int in_width2 = in_width0 + 2;
    int in_width3 = in_width0 + 3;
    int in_width4 = in_width0 + 4;
    int in_width5 = in_width0 + 5;

    const int height_start = out_height_idx - padding_wh.y;
    int in_height_start = select(height_start, 0, height_start < 0);
    int in_height_end = min(3 + height_start, input_wh.y);

    const int batch_idx = mul24(out_batch_idx, input_wh.y);
    const int weights_h_idx_s0 = mad24(out_channel_block_idx, 9,
                                    mul24(in_height_start - height_start, 3));
    const int weights_h_idx_s1 = weights_h_idx_s0 + 9;

    FLOAT4 in0, in1, in2, in3, in4, in5;
    FLOAT4 weights_w0_c0_s0, weights_w0_c1_s0, weights_w0_c2_s0, weights_w0_c3_s0;
    FLOAT4 weights_w1_c0_s0, weights_w1_c1_s0, weights_w1_c2_s0, weights_w1_c3_s0;
    FLOAT4 weights_w2_c0_s0, weights_w2_c1_s0, weights_w2_c2_s0, weights_w2_c3_s0;
    FLOAT4 weights_w0_c0_s1, weights_w0_c1_s1, weights_w0_c2_s1, weights_w0_c3_s1;
    FLOAT4 weights_w1_c0_s1, weights_w1_c1_s1, weights_w1_c2_s1, weights_w1_c3_s1;
    FLOAT4 weights_w2_c0_s1, weights_w2_c1_s1, weights_w2_c2_s1, weights_w2_c3_s1;
    int weights_y_idx_s0 = weights_h_idx_s0;
    int weights_y_idx_s1 = weights_h_idx_s1;
    for (int iy = in_height_start; iy < in_height_end; iy++) {
        int in_hb_value = iy + batch_idx;
        for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
            const int in_idx  = mul24(input_c_block_idx, input_wh.x);
            int weights_x_idx = input_c_block_idx << 2;
            READ_INPUT_IMAGE(0, 0);
            READ_INPUT_IMAGE(1, 0);
            READ_INPUT_IMAGE(2, 0);
            READ_INPUT_IMAGE(3, 0);
            READ_INPUT_IMAGE(4, 0);
            READ_INPUT_IMAGE(5, 0);

            weights_w0_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
            weights_w0_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s0));
            weights_w0_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s0));
            weights_w0_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s0));

            weights_w1_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0 + 1));
            weights_w1_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s0 + 1));
            weights_w1_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s0 + 1));
            weights_w1_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s0 + 1));

            weights_w2_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0 + 2));
            weights_w2_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s0 + 2));
            weights_w2_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s0 + 2));
            weights_w2_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s0 + 2));

#if 0
            out_w0_s0 = mad(in0.x, weights_w0_c0_s0, out_w0_s0);
            out_w0_s0 = mad(in0.y, weights_w0_c1_s0, out_w0_s0);
            out_w0_s0 = mad(in0.z, weights_w0_c2_s0, out_w0_s0);
            out_w0_s0 = mad(in0.w, weights_w0_c3_s0, out_w0_s0);
            out_w0_s0 = mad(in1.x, weights_w1_c0_s0, out_w0_s0);
            out_w0_s0 = mad(in1.y, weights_w1_c1_s0, out_w0_s0);
            out_w0_s0 = mad(in1.z, weights_w1_c2_s0, out_w0_s0);
            out_w0_s0 = mad(in1.w, weights_w1_c3_s0, out_w0_s0);
            out_w0_s0 = mad(in2.x, weights_w2_c0_s0, out_w0_s0);
            out_w0_s0 = mad(in2.y, weights_w2_c1_s0, out_w0_s0);
            out_w0_s0 = mad(in2.z, weights_w2_c2_s0, out_w0_s0);
            out_w0_s0 = mad(in2.w, weights_w2_c3_s0, out_w0_s0);

            out_w1_s0 = mad(in1.x, weights_w0_c0_s0, out_w1_s0);
            out_w1_s0 = mad(in1.y, weights_w0_c1_s0, out_w1_s0);
            out_w1_s0 = mad(in1.z, weights_w0_c2_s0, out_w1_s0);
            out_w1_s0 = mad(in1.w, weights_w0_c3_s0, out_w1_s0);
            out_w1_s0 = mad(in2.x, weights_w1_c0_s0, out_w1_s0);
            out_w1_s0 = mad(in2.y, weights_w1_c1_s0, out_w1_s0);
            out_w1_s0 = mad(in2.z, weights_w1_c2_s0, out_w1_s0);
            out_w1_s0 = mad(in2.w, weights_w1_c3_s0, out_w1_s0);
            out_w1_s0 = mad(in3.x, weights_w2_c0_s0, out_w1_s0);
            out_w1_s0 = mad(in3.y, weights_w2_c1_s0, out_w1_s0);
            out_w1_s0 = mad(in3.z, weights_w2_c2_s0, out_w1_s0);
            out_w1_s0 = mad(in3.w, weights_w2_c3_s0, out_w1_s0);

            out_w2_s0 = mad(in2.x, weights_w0_c0_s0, out_w2_s0);
            out_w2_s0 = mad(in2.y, weights_w0_c1_s0, out_w2_s0);
            out_w2_s0 = mad(in2.z, weights_w0_c2_s0, out_w2_s0);
            out_w2_s0 = mad(in2.w, weights_w0_c3_s0, out_w2_s0);
            out_w2_s0 = mad(in3.x, weights_w1_c0_s0, out_w2_s0);
            out_w2_s0 = mad(in3.y, weights_w1_c1_s0, out_w2_s0);
            out_w2_s0 = mad(in3.z, weights_w1_c2_s0, out_w2_s0);
            out_w2_s0 = mad(in3.w, weights_w1_c3_s0, out_w2_s0);
            out_w2_s0 = mad(in4.x, weights_w2_c0_s0, out_w2_s0);
            out_w2_s0 = mad(in4.y, weights_w2_c1_s0, out_w2_s0);
            out_w2_s0 = mad(in4.z, weights_w2_c2_s0, out_w2_s0);
            out_w2_s0 = mad(in4.w, weights_w2_c3_s0, out_w2_s0);

            out_w3_s0 = mad(in3.x, weights_w0_c0_s0, out_w3_s0);
            out_w3_s0 = mad(in3.y, weights_w0_c1_s0, out_w3_s0);
            out_w3_s0 = mad(in3.z, weights_w0_c2_s0, out_w3_s0);
            out_w3_s0 = mad(in3.w, weights_w0_c3_s0, out_w3_s0);
            out_w3_s0 = mad(in4.x, weights_w1_c0_s0, out_w3_s0);
            out_w3_s0 = mad(in4.y, weights_w1_c1_s0, out_w3_s0);
            out_w3_s0 = mad(in4.z, weights_w1_c2_s0, out_w3_s0);
            out_w3_s0 = mad(in4.w, weights_w1_c3_s0, out_w3_s0);
            out_w3_s0 = mad(in5.x, weights_w2_c0_s0, out_w3_s0);
            out_w3_s0 = mad(in5.y, weights_w2_c1_s0, out_w3_s0);
            out_w3_s0 = mad(in5.z, weights_w2_c2_s0, out_w3_s0);
            out_w3_s0 = mad(in5.w, weights_w2_c3_s0, out_w3_s0);

            if (is_s1_in_boundary) {
                weights_w0_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_w0_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1));
                weights_w0_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1));
                weights_w0_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1));

                weights_w1_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1 + 1));
                weights_w1_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1 + 1));
                weights_w1_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1 + 1));
                weights_w1_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1 + 1));

                weights_w2_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1 + 2));
                weights_w2_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1 + 2));
                weights_w2_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1 + 2));
                weights_w2_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1 + 2));

                out_w0_s1 = mad(in0.x, weights_w0_c0_s1, out_w0_s1);
                out_w0_s1 = mad(in0.y, weights_w0_c1_s1, out_w0_s1);
                out_w0_s1 = mad(in0.z, weights_w0_c2_s1, out_w0_s1);
                out_w0_s1 = mad(in0.w, weights_w0_c3_s1, out_w0_s1);
                out_w0_s1 = mad(in1.x, weights_w1_c0_s1, out_w0_s1);
                out_w0_s1 = mad(in1.y, weights_w1_c1_s1, out_w0_s1);
                out_w0_s1 = mad(in1.z, weights_w1_c2_s1, out_w0_s1);
                out_w0_s1 = mad(in1.w, weights_w1_c3_s1, out_w0_s1);
                out_w0_s1 = mad(in2.x, weights_w2_c0_s1, out_w0_s1);
                out_w0_s1 = mad(in2.y, weights_w2_c1_s1, out_w0_s1);
                out_w0_s1 = mad(in2.z, weights_w2_c2_s1, out_w0_s1);
                out_w0_s1 = mad(in2.w, weights_w2_c3_s1, out_w0_s1);

                out_w1_s1 = mad(in1.x, weights_w0_c0_s1, out_w1_s1);
                out_w1_s1 = mad(in1.y, weights_w0_c1_s1, out_w1_s1);
                out_w1_s1 = mad(in1.z, weights_w0_c2_s1, out_w1_s1);
                out_w1_s1 = mad(in1.w, weights_w0_c3_s1, out_w1_s1);
                out_w1_s1 = mad(in2.x, weights_w1_c0_s1, out_w1_s1);
                out_w1_s1 = mad(in2.y, weights_w1_c1_s1, out_w1_s1);
                out_w1_s1 = mad(in2.z, weights_w1_c2_s1, out_w1_s1);
                out_w1_s1 = mad(in2.w, weights_w1_c3_s1, out_w1_s1);
                out_w1_s1 = mad(in3.x, weights_w2_c0_s1, out_w1_s1);
                out_w1_s1 = mad(in3.y, weights_w2_c1_s1, out_w1_s1);
                out_w1_s1 = mad(in3.z, weights_w2_c2_s1, out_w1_s1);
                out_w1_s1 = mad(in3.w, weights_w2_c3_s1, out_w1_s1);

                out_w2_s1 = mad(in2.x, weights_w0_c0_s1, out_w2_s1);
                out_w2_s1 = mad(in2.y, weights_w0_c1_s1, out_w2_s1);
                out_w2_s1 = mad(in2.z, weights_w0_c2_s1, out_w2_s1);
                out_w2_s1 = mad(in2.w, weights_w0_c3_s1, out_w2_s1);
                out_w2_s1 = mad(in3.x, weights_w1_c0_s1, out_w2_s1);
                out_w2_s1 = mad(in3.y, weights_w1_c1_s1, out_w2_s1);
                out_w2_s1 = mad(in3.z, weights_w1_c2_s1, out_w2_s1);
                out_w2_s1 = mad(in3.w, weights_w1_c3_s1, out_w2_s1);
                out_w2_s1 = mad(in4.x, weights_w2_c0_s1, out_w2_s1);
                out_w2_s1 = mad(in4.y, weights_w2_c1_s1, out_w2_s1);
                out_w2_s1 = mad(in4.z, weights_w2_c2_s1, out_w2_s1);
                out_w2_s1 = mad(in4.w, weights_w2_c3_s1, out_w2_s1);

                out_w3_s1 = mad(in3.x, weights_w0_c0_s1, out_w3_s1);
                out_w3_s1 = mad(in3.y, weights_w0_c1_s1, out_w3_s1);
                out_w3_s1 = mad(in3.z, weights_w0_c2_s1, out_w3_s1);
                out_w3_s1 = mad(in3.w, weights_w0_c3_s1, out_w3_s1);
                out_w3_s1 = mad(in4.x, weights_w1_c0_s1, out_w3_s1);
                out_w3_s1 = mad(in4.y, weights_w1_c1_s1, out_w3_s1);
                out_w3_s1 = mad(in4.z, weights_w1_c2_s1, out_w3_s1);
                out_w3_s1 = mad(in4.w, weights_w1_c3_s1, out_w3_s1);
                out_w3_s1 = mad(in5.x, weights_w2_c0_s1, out_w3_s1);
                out_w3_s1 = mad(in5.y, weights_w2_c1_s1, out_w3_s1);
                out_w3_s1 = mad(in5.z, weights_w2_c2_s1, out_w3_s1);
                out_w3_s1 = mad(in5.w, weights_w2_c3_s1, out_w3_s1);
            }
#else
            out_w0_s0 += in0.x * weights_w0_c0_s0;
            out_w0_s0 += in0.y * weights_w0_c1_s0;
            out_w0_s0 += in0.z * weights_w0_c2_s0;
            out_w0_s0 += in0.w * weights_w0_c3_s0;
            out_w0_s0 += in1.x * weights_w1_c0_s0;
            out_w0_s0 += in1.y * weights_w1_c1_s0;
            out_w0_s0 += in1.z * weights_w1_c2_s0;
            out_w0_s0 += in1.w * weights_w1_c3_s0;
            out_w0_s0 += in2.x * weights_w2_c0_s0;
            out_w0_s0 += in2.y * weights_w2_c1_s0;
            out_w0_s0 += in2.z * weights_w2_c2_s0;
            out_w0_s0 += in2.w * weights_w2_c3_s0;

            out_w1_s0 += in1.x * weights_w0_c0_s0;
            out_w1_s0 += in1.y * weights_w0_c1_s0;
            out_w1_s0 += in1.z * weights_w0_c2_s0;
            out_w1_s0 += in1.w * weights_w0_c3_s0;
            out_w1_s0 += in2.x * weights_w1_c0_s0;
            out_w1_s0 += in2.y * weights_w1_c1_s0;
            out_w1_s0 += in2.z * weights_w1_c2_s0;
            out_w1_s0 += in2.w * weights_w1_c3_s0;
            out_w1_s0 += in3.x * weights_w2_c0_s0;
            out_w1_s0 += in3.y * weights_w2_c1_s0;
            out_w1_s0 += in3.z * weights_w2_c2_s0;
            out_w1_s0 += in3.w * weights_w2_c3_s0;

            out_w2_s0 += in2.x * weights_w0_c0_s0;
            out_w2_s0 += in2.y * weights_w0_c1_s0;
            out_w2_s0 += in2.z * weights_w0_c2_s0;
            out_w2_s0 += in2.w * weights_w0_c3_s0;
            out_w2_s0 += in3.x * weights_w1_c0_s0;
            out_w2_s0 += in3.y * weights_w1_c1_s0;
            out_w2_s0 += in3.z * weights_w1_c2_s0;
            out_w2_s0 += in3.w * weights_w1_c3_s0;
            out_w2_s0 += in4.x * weights_w2_c0_s0;
            out_w2_s0 += in4.y * weights_w2_c1_s0;
            out_w2_s0 += in4.z * weights_w2_c2_s0;
            out_w2_s0 += in4.w * weights_w2_c3_s0;

            out_w3_s0 += in3.x * weights_w0_c0_s0;
            out_w3_s0 += in3.y * weights_w0_c1_s0;
            out_w3_s0 += in3.z * weights_w0_c2_s0;
            out_w3_s0 += in3.w * weights_w0_c3_s0;
            out_w3_s0 += in4.x * weights_w1_c0_s0;
            out_w3_s0 += in4.y * weights_w1_c1_s0;
            out_w3_s0 += in4.z * weights_w1_c2_s0;
            out_w3_s0 += in4.w * weights_w1_c3_s0;
            out_w3_s0 += in5.x * weights_w2_c0_s0;
            out_w3_s0 += in5.y * weights_w2_c1_s0;
            out_w3_s0 += in5.z * weights_w2_c2_s0;
            out_w3_s0 += in5.w * weights_w2_c3_s0;

            if (is_s1_in_boundary) {
                weights_w0_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_w0_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1));
                weights_w0_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1));
                weights_w0_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1));

                weights_w1_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1 + 1));
                weights_w1_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1 + 1));
                weights_w1_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1 + 1));
                weights_w1_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1 + 1));

                weights_w2_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1 + 2));
                weights_w2_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1 + 2));
                weights_w2_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1 + 2));
                weights_w2_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1 + 2));

                out_w0_s1 += in0.x * weights_w0_c0_s1;
                out_w0_s1 += in0.y * weights_w0_c1_s1;
                out_w0_s1 += in0.z * weights_w0_c2_s1;
                out_w0_s1 += in0.w * weights_w0_c3_s1;
                out_w0_s1 += in1.x * weights_w1_c0_s1;
                out_w0_s1 += in1.y * weights_w1_c1_s1;
                out_w0_s1 += in1.z * weights_w1_c2_s1;
                out_w0_s1 += in1.w * weights_w1_c3_s1;
                out_w0_s1 += in2.x * weights_w2_c0_s1;
                out_w0_s1 += in2.y * weights_w2_c1_s1;
                out_w0_s1 += in2.z * weights_w2_c2_s1;
                out_w0_s1 += in2.w * weights_w2_c3_s1;

                out_w1_s1 += in1.x * weights_w0_c0_s1;
                out_w1_s1 += in1.y * weights_w0_c1_s1;
                out_w1_s1 += in1.z * weights_w0_c2_s1;
                out_w1_s1 += in1.w * weights_w0_c3_s1;
                out_w1_s1 += in2.x * weights_w1_c0_s1;
                out_w1_s1 += in2.y * weights_w1_c1_s1;
                out_w1_s1 += in2.z * weights_w1_c2_s1;
                out_w1_s1 += in2.w * weights_w1_c3_s1;
                out_w1_s1 += in3.x * weights_w2_c0_s1;
                out_w1_s1 += in3.y * weights_w2_c1_s1;
                out_w1_s1 += in3.z * weights_w2_c2_s1;
                out_w1_s1 += in3.w * weights_w2_c3_s1;

                out_w2_s1 += in2.x * weights_w0_c0_s1;
                out_w2_s1 += in2.y * weights_w0_c1_s1;
                out_w2_s1 += in2.z * weights_w0_c2_s1;
                out_w2_s1 += in2.w * weights_w0_c3_s1;
                out_w2_s1 += in3.x * weights_w1_c0_s1;
                out_w2_s1 += in3.y * weights_w1_c1_s1;
                out_w2_s1 += in3.z * weights_w1_c2_s1;
                out_w2_s1 += in3.w * weights_w1_c3_s1;
                out_w2_s1 += in4.x * weights_w2_c0_s1;
                out_w2_s1 += in4.y * weights_w2_c1_s1;
                out_w2_s1 += in4.z * weights_w2_c2_s1;
                out_w2_s1 += in4.w * weights_w2_c3_s1;

                out_w3_s1 += in3.x * weights_w0_c0_s1;
                out_w3_s1 += in3.y * weights_w0_c1_s1;
                out_w3_s1 += in3.z * weights_w0_c2_s1;
                out_w3_s1 += in3.w * weights_w0_c3_s1;
                out_w3_s1 += in4.x * weights_w1_c0_s1;
                out_w3_s1 += in4.y * weights_w1_c1_s1;
                out_w3_s1 += in4.z * weights_w1_c2_s1;
                out_w3_s1 += in4.w * weights_w1_c3_s1;
                out_w3_s1 += in5.x * weights_w2_c0_s1;
                out_w3_s1 += in5.y * weights_w2_c1_s1;
                out_w3_s1 += in5.z * weights_w2_c2_s1;
                out_w3_s1 += in5.w * weights_w2_c3_s1;
            }
#endif
        }
        weights_y_idx_s0 += 3;
        weights_y_idx_s1 += 3;
    }

    out_w0_s0 = ActivationProcess(out_w0_s0);
    out_w1_s0 = ActivationProcess(out_w1_s0);
    out_w2_s0 = ActivationProcess(out_w2_s0);
    out_w3_s0 = ActivationProcess(out_w3_s0);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    #if 1
    WriteOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0, out_w2_s0, out_w3_s0, output_w_idx,
                               output_bh_idx, remain);
    #else
    if (out_channel_block_idx == 0 && output_w_idx == 0 && output_bh_idx == 0) {
        WI_F(output, (int2)(0, 0), out_w0_s0 + 128);
    }
    #endif

    if (is_s1_in_boundary) {
        out_w0_s1 = ActivationProcess(out_w0_s1);
        out_w1_s1 = ActivationProcess(out_w1_s1);
        out_w2_s1 = ActivationProcess(out_w2_s1);
        out_w3_s1 = ActivationProcess(out_w3_s1);

        output_w_idx += output_wh.x;
        WriteOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1, out_w2_s1, out_w3_s1, output_w_idx,
                               output_bh_idx, remain);
    }
}

__kernel void Conv2DGS3D(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);

                weights0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx));
                weights1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx));
                weights2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx));
                weights3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx++));

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void Conv2DGS3Dcb2(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
#if 1
    __read_only image2d_t weights,
#else
    __read_only image2d_t weights0,
    __read_only image2d_t weights1,
    __read_only image2d_t weights2,
    __read_only image2d_t weights3, 
#endif
    __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int out_channel_block_length,
    __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks,
    int4 shared_int4_0,
    int4 shared_int4_1,
    int4 shared_int4_2) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    #if 1
    const int out_channel_slice_idx = get_global_id(0);
    const int DST_S = out_channel_slice_idx << 1;
    const int out_width_block_idx   = get_global_id(1);
    const int DST_Y         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_slice_idx, out_width_block_idx, DST_Y);
    #else
    const int out_channel_slice_idx = get_global_id(2);
    const int DST_S = out_channel_slice_idx << 1;
    const int out_width_block_idx   = get_global_id(0);
    const int DST_X = out_width_block_idx << 2;
    const int DST_Y         = get_global_id(1);
    DEAL_NON_UNIFORM_DIM3(out_width_block_idx, DST_Y, out_channel_slice_idx);
    #endif

    const int out_batch_idx  = DST_Y / output_wh.y;
    const int out_height_idx = DST_Y % output_wh.y;
    bool is_s1_in_boundary = (DST_S + 1 < out_channel_block_length);

    FLOAT4 out_w0_s0 = RI_F(bias, SAMPLER, (int2)(DST_S, 0));
    // FLOAT4 out_w0_s0 = 0.0f;
    FLOAT4 out_w1_s0 = out_w0_s0;
    FLOAT4 out_w2_s0 = out_w0_s0;
    FLOAT4 out_w3_s0 = out_w0_s0;

#if 1
    FLOAT4 out_w0_s1 = RI_F(bias, SAMPLER, (int2)(DST_S + 1, 0));
    // FLOAT4 out_w0_s1 = 0.0f;
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;
#endif

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((DST_Y % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((DST_Y / output_wh.y), input_wh.y);
    const int weights_h_idx_s0 = mul24(DST_S, mul24(kernel_wh.x, kernel_wh.y)) +
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);
    const int weights_h_idx_s1 = weights_h_idx_s0 + mul24(kernel_wh.x, kernel_wh.y);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights_c0_s0, weights_c1_s0, weights_c2_s0, weights_c3_s0;
    FLOAT4 weights_c0_s1, weights_c1_s1, weights_c2_s1, weights_c3_s1;
    
    int weights_y_idx_s0 = weights_h_idx_s0;
    int weights_y_idx_s1 = weights_h_idx_s1;
    int filter_offset = 0;
    for (int ky = 0; ky < kernel_wh.y; ++ky) {
        int in_hb_value = mul24(ky, dilation_wh.y) + height_start;
        int yck0 = mul24(ky, dilation_wh.y) + height_start;
        for (int w = 0; w < kernel_wh.x; w++) {
            int input_w_base = mul24(w, dilation_wh.x);
            int s = 0;
            int xck0 = input_w_base + in_width0;
            int xck1 = input_w_base + in_width1;
            int xck2 = input_w_base + in_width2;
            int xck3 = input_w_base + in_width3;
            do {
                const int in_idx  = mul24(s, input_wh.x);
                int weights_x_idx = s << 2;
                
                #if 1
                weights_c0_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
                weights_c1_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s0));
                weights_c2_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s0));
                weights_c3_s0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s0));
                #elif 1
                weights_c0_s0 = RI_F(weights0, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
                weights_c1_s0 = RI_F(weights1, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
                weights_c2_s0 = RI_F(weights2, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
                weights_c3_s0 = RI_F(weights3, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s0));
                #elif 1
                weights_c0_s0 = RI_F(weights0, smp_none, (int2)(DST_S, filter_offset));
                weights_c1_s0 = RI_F(weights1, smp_none, (int2)(DST_S, filter_offset));
                weights_c2_s0 = RI_F(weights2, smp_none, (int2)(DST_S, filter_offset));
                weights_c3_s0 = RI_F(weights3, smp_none, (int2)(DST_S, filter_offset));
                #endif

#if 1
                #if 1
                weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx_s1));
                weights_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx_s1));
                weights_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx_s1));
                #elif 1
                weights_c0_s1 = RI_F(weights0, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_c1_s1 = RI_F(weights1, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_c2_s1 = RI_F(weights2, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                weights_c3_s1 = RI_F(weights3, SAMPLER, (int2)(weights_x_idx, weights_y_idx_s1));
                #elif 1
                weights_c0_s1 = RI_F(weights0, smp_none, (int2)(DST_S + 1, filter_offset));
                weights_c1_s1 = RI_F(weights1, smp_none, (int2)(DST_S + 1, filter_offset));
                weights_c2_s1 = RI_F(weights2, smp_none, (int2)(DST_S + 1, filter_offset));
                weights_c3_s1 = RI_F(weights3, smp_none, (int2)(DST_S + 1, filter_offset));
                #endif
                filter_offset++;
#elif 1
                weights_c0_s1 = weights_c0_s0;
                weights_c1_s1 = weights_c0_s0;
                weights_c2_s1 = weights_c0_s0;
                weights_c3_s1 = weights_c0_s0;
#else
                weights_c0_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 4, weights_y_idx_s0));
                weights_c1_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 4 + 1, weights_y_idx_s0));
                weights_c2_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 4 + 2, weights_y_idx_s0));
                weights_c3_s1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 4 + 3, weights_y_idx_s0));
#endif

                #if 1
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);
                #elif 0
                in0 = read_imageh(input, smp_zero, (int2)((s * input_wh.x + input_w_base + in_width0), in_hb_value));
                in1 = read_imageh(input, smp_zero, (int2)((s * input_wh.x + input_w_base + in_width1), in_hb_value));
                in2 = read_imageh(input, smp_zero, (int2)((s * input_wh.x + input_w_base + in_width2), in_hb_value));
                in3 = read_imageh(input, smp_zero, (int2)((s * input_wh.x + input_w_base + in_width3), in_hb_value));
                #elif 1
                in0 = read_imageh(input, smp_zero, (int2)(xck0, (yck0) * in_channel_block_length + (s)));
                in1 = read_imageh(input, smp_zero, (int2)(xck1, (yck0) * in_channel_block_length + (s)));
                in2 = read_imageh(input, smp_zero, (int2)(xck2, (yck0) * in_channel_block_length + (s)));
                in3 = read_imageh(input, smp_zero, (int2)(xck3, (yck0) * in_channel_block_length + (s)));
                #endif

                s += 1;
#if 0
                out_w0_s0 = mad(in0.x, weights_c0_s0, out_w0_s0);
                out_w0_s0 = mad(in0.y, weights_c1_s0, out_w0_s0);
                out_w0_s0 = mad(in0.z, weights_c2_s0, out_w0_s0);
                out_w0_s0 = mad(in0.w, weights_c3_s0, out_w0_s0);

                out_w1_s0 = mad(in1.x, weights_c0_s0, out_w1_s0);
                out_w1_s0 = mad(in1.y, weights_c1_s0, out_w1_s0);
                out_w1_s0 = mad(in1.z, weights_c2_s0, out_w1_s0);
                out_w1_s0 = mad(in1.w, weights_c3_s0, out_w1_s0);

                out_w2_s0 = mad(in2.x, weights_c0_s0, out_w2_s0);
                out_w2_s0 = mad(in2.y, weights_c1_s0, out_w2_s0);
                out_w2_s0 = mad(in2.z, weights_c2_s0, out_w2_s0);
                out_w2_s0 = mad(in2.w, weights_c3_s0, out_w2_s0);

                out_w3_s0 = mad(in3.x, weights_c0_s0, out_w3_s0);
                out_w3_s0 = mad(in3.y, weights_c1_s0, out_w3_s0);
                out_w3_s0 = mad(in3.z, weights_c2_s0, out_w3_s0);
                out_w3_s0 = mad(in3.w, weights_c3_s0, out_w3_s0);
                
#if 1
                out_w0_s1 = mad(in0.x, weights_c0_s1, out_w0_s1);
                out_w0_s1 = mad(in0.y, weights_c1_s1, out_w0_s1);
                out_w0_s1 = mad(in0.z, weights_c2_s1, out_w0_s1);
                out_w0_s1 = mad(in0.w, weights_c3_s1, out_w0_s1);

                out_w1_s1 = mad(in1.x, weights_c0_s1, out_w1_s1);
                out_w1_s1 = mad(in1.y, weights_c1_s1, out_w1_s1);
                out_w1_s1 = mad(in1.z, weights_c2_s1, out_w1_s1);
                out_w1_s1 = mad(in1.w, weights_c3_s1, out_w1_s1);

                out_w2_s1 = mad(in2.x, weights_c0_s1, out_w2_s1);
                out_w2_s1 = mad(in2.y, weights_c1_s1, out_w2_s1);
                out_w2_s1 = mad(in2.z, weights_c2_s1, out_w2_s1);
                out_w2_s1 = mad(in2.w, weights_c3_s1, out_w2_s1);

                out_w3_s1 = mad(in3.x, weights_c0_s1, out_w3_s1);
                out_w3_s1 = mad(in3.y, weights_c1_s1, out_w3_s1);
                out_w3_s1 = mad(in3.z, weights_c2_s1, out_w3_s1);
                out_w3_s1 = mad(in3.w, weights_c3_s1, out_w3_s1);
#else
#if 1
                out_w0_s0 = mad(in0.x, weights_c0_s1, out_w0_s0);
                out_w0_s0 = mad(in0.y, weights_c1_s1, out_w0_s0);
                out_w0_s0 = mad(in0.z, weights_c2_s1, out_w0_s0);
                out_w0_s0 = mad(in0.w, weights_c3_s1, out_w0_s0);

                out_w1_s0 = mad(in1.x, weights_c0_s1, out_w1_s0);
                out_w1_s0 = mad(in1.y, weights_c1_s1, out_w1_s0);
                out_w1_s0 = mad(in1.z, weights_c2_s1, out_w1_s0);
                out_w1_s0 = mad(in1.w, weights_c3_s1, out_w1_s0);

                out_w2_s0 = mad(in2.x, weights_c0_s1, out_w2_s0);
                out_w2_s0 = mad(in2.y, weights_c1_s1, out_w2_s0);
                out_w2_s0 = mad(in2.z, weights_c2_s1, out_w2_s0);
                out_w2_s0 = mad(in2.w, weights_c3_s1, out_w2_s0);

                out_w3_s0 = mad(in3.x, weights_c0_s1, out_w3_s0);
                out_w3_s0 = mad(in3.y, weights_c1_s1, out_w3_s0);
                out_w3_s0 = mad(in3.z, weights_c2_s1, out_w3_s0);
                out_w3_s0 = mad(in3.w, weights_c3_s1, out_w3_s0);
#endif
#endif
#else
#if 0
                out_w0_s0 += in0.x * weights_c0_s0;
                out_w0_s0 += in0.y * weights_c1_s0;
                out_w0_s0 += in0.z * weights_c2_s0;
                out_w0_s0 += in0.w * weights_c3_s0;
                out_w1_s0 += in1.x * weights_c0_s0;
                out_w1_s0 += in1.y * weights_c1_s0;
                out_w1_s0 += in1.z * weights_c2_s0;
                out_w1_s0 += in1.w * weights_c3_s0;
                out_w2_s0 += in2.x * weights_c0_s0;
                out_w2_s0 += in2.y * weights_c1_s0;
                out_w2_s0 += in2.z * weights_c2_s0;
                out_w2_s0 += in2.w * weights_c3_s0;
                out_w3_s0 += in3.x * weights_c0_s0;
                out_w3_s0 += in3.y * weights_c1_s0;
                out_w3_s0 += in3.z * weights_c2_s0;
                out_w3_s0 += in3.w * weights_c3_s0;
                out_w0_s1 += in0.x * weights_c0_s1;
                out_w0_s1 += in0.y * weights_c1_s1;
                out_w0_s1 += in0.z * weights_c2_s1;
                out_w0_s1 += in0.w * weights_c3_s1;
                out_w1_s1 += in1.x * weights_c0_s1;
                out_w1_s1 += in1.y * weights_c1_s1;
                out_w1_s1 += in1.z * weights_c2_s1;
                out_w1_s1 += in1.w * weights_c3_s1;
                out_w2_s1 += in2.x * weights_c0_s1;
                out_w2_s1 += in2.y * weights_c1_s1;
                out_w2_s1 += in2.z * weights_c2_s1;
                out_w2_s1 += in2.w * weights_c3_s1;
                out_w3_s1 += in3.x * weights_c0_s1;
                out_w3_s1 += in3.y * weights_c1_s1;
                out_w3_s1 += in3.z * weights_c2_s1;
                out_w3_s1 += in3.w * weights_c3_s1;
#elif 0
                out_w0_s0 += weights_c0_s0 * in0.x;
                out_w1_s0 += weights_c0_s0 * in1.x;
                out_w2_s0 += weights_c0_s0 * in2.x;
                out_w3_s0 += weights_c0_s0 * in3.x;
                out_w0_s0 += weights_c1_s0 * in0.y;
                out_w1_s0 += weights_c1_s0 * in1.y;
                out_w2_s0 += weights_c1_s0 * in2.y;
                out_w3_s0 += weights_c1_s0 * in3.y;
                out_w0_s0 += weights_c2_s0 * in0.z;
                out_w1_s0 += weights_c2_s0 * in1.z;
                out_w2_s0 += weights_c2_s0 * in2.z;
                out_w3_s0 += weights_c2_s0 * in3.z;
                out_w0_s0 += weights_c3_s0 * in0.w;
                out_w1_s0 += weights_c3_s0 * in1.w;
                out_w2_s0 += weights_c3_s0 * in2.w;
                out_w3_s0 += weights_c3_s0 * in3.w;
                out_w0_s1 += weights_c0_s1 * in0.x;
                out_w1_s1 += weights_c0_s1 * in1.x;
                out_w2_s1 += weights_c0_s1 * in2.x;
                out_w3_s1 += weights_c0_s1 * in3.x;
                out_w0_s1 += weights_c1_s1 * in0.y;
                out_w1_s1 += weights_c1_s1 * in1.y;
                out_w2_s1 += weights_c1_s1 * in2.y;
                out_w3_s1 += weights_c1_s1 * in3.y;
                out_w0_s1 += weights_c2_s1 * in0.z;
                out_w1_s1 += weights_c2_s1 * in1.z;
                out_w2_s1 += weights_c2_s1 * in2.z;
                out_w3_s1 += weights_c2_s1 * in3.z;
                out_w0_s1 += weights_c3_s1 * in0.w;
                out_w1_s1 += weights_c3_s1 * in1.w;
                out_w2_s1 += weights_c3_s1 * in2.w;
                out_w3_s1 += weights_c3_s1 * in3.w;
#elif 1
                out_w0_s0 += weights_c0_s0 * in0.x;
                out_w1_s0 += weights_c0_s0 * in1.x;
                out_w2_s0 += weights_c0_s0 * in2.x;
                out_w3_s0 += weights_c0_s0 * in3.x;
                out_w0_s0 += weights_c1_s0 * in0.y;
                out_w1_s0 += weights_c1_s0 * in1.y;
                out_w2_s0 += weights_c1_s0 * in2.y;
                out_w3_s0 += weights_c1_s0 * in3.y;
                out_w0_s0 += weights_c2_s0 * in0.z;
                out_w1_s0 += weights_c2_s0 * in1.z;
                out_w2_s0 += weights_c2_s0 * in2.z;
                out_w3_s0 += weights_c2_s0 * in3.z;
                out_w0_s0 += weights_c3_s0 * in0.w;
                out_w1_s0 += weights_c3_s0 * in1.w;
                out_w2_s0 += weights_c3_s0 * in2.w;
                out_w3_s0 += weights_c3_s0 * in3.w;
                out_w0_s1 += weights_c0_s1 * in0.x;
                out_w1_s1 += weights_c0_s1 * in1.x;
                out_w2_s1 += weights_c0_s1 * in2.x;
                out_w3_s1 += weights_c0_s1 * in3.x;
                out_w0_s1 += weights_c1_s1 * in0.y;
                out_w1_s1 += weights_c1_s1 * in1.y;
                out_w2_s1 += weights_c1_s1 * in2.y;
                out_w3_s1 += weights_c1_s1 * in3.y;
                out_w0_s1 += weights_c2_s1 * in0.z;
                out_w1_s1 += weights_c2_s1 * in1.z;
                out_w2_s1 += weights_c2_s1 * in2.z;
                out_w3_s1 += weights_c2_s1 * in3.z;
                out_w0_s1 += weights_c3_s1 * in0.w;
                out_w1_s1 += weights_c3_s1 * in1.w;
                out_w2_s1 += weights_c3_s1 * in2.w;
                out_w3_s1 += weights_c3_s1 * in3.w;
#endif
#endif
            } while(s < in_channel_block_length);
            weights_y_idx_s0++;
            weights_y_idx_s1++;
        }
    }

#if 1
    out_w0_s0 = ActivationProcess(out_w0_s0);
    out_w1_s0 = ActivationProcess(out_w1_s0);
    out_w2_s0 = ActivationProcess(out_w2_s0);
    out_w3_s0 = ActivationProcess(out_w3_s0);

#if 1
    out_w0_s1 = ActivationProcess(out_w0_s1);
    out_w1_s1 = ActivationProcess(out_w1_s1);
    out_w2_s1 = ActivationProcess(out_w2_s1);
    out_w3_s1 = ActivationProcess(out_w3_s1);
#endif

#if 0
    FLOAT4 out_w0_s1 = 0.0f;
    FLOAT4 out_w1_s1 = out_w0_s1;
    FLOAT4 out_w2_s1 = out_w0_s1;
    FLOAT4 out_w3_s1 = out_w0_s1;
#endif

    const int out_x_base = mul24(DST_S, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
#if 0
    WriteOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0, out_w2_s0, out_w3_s0, output_w_idx,
                               DST_Y, remain);

#elif 1
    WriteOutputAntiOutOfBounds(output, out_w0_s0, out_w1_s0, out_w2_s0, out_w3_s0, output_w_idx,
                               DST_Y, remain);
    
    output_w_idx += output_wh.x;
    WriteOutputAntiOutOfBounds(output, out_w0_s1, out_w1_s1, out_w2_s1, out_w3_s1, output_w_idx,
                               DST_Y, remain);
#elif 0
    #if 0
    if (remain >= 4) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w2_s0);
        write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w3_s0);
    } else if (remain == 3) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w2_s0);
    } else if (remain == 2) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
    } else if (remain == 1) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
    }
    #else
    if (DST_S + 0 >= shared_int4_0.z) return;
    FLOAT4 bias_val = 0.0f;
  {
    FLOAT4 res = out_w0_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
    // write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
  }
#if 1
    // if (DST_X + 1 < shared_int4_0.x) {
    if (remain >= 2) {
    write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
  }
#else
    write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
#endif
    if (remain >= 3) {
    write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w2_s0);
    }
    if (remain >= 4) {
    write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w3_s0);
    }
    #endif

    output_w_idx += output_wh.x;

    #if 0
    if (remain >= 4) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w1_s1);
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w2_s1);
        write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w3_s1);
    } else if (remain == 3) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w1_s1);
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w2_s1);
    } else if (remain == 2) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w1_s1);
    } else if (remain == 1) {
        write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
    }
    #elif 1
    write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
    if (remain >= 4) {
        write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w3_s1);
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w2_s1);
    } else if (remain >= 3) {
        write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w2_s1);
    }
    if (remain >= 2) {
        write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w1_s1);
    }
    
    
    #endif
#else
    if (DST_S + 0 >= shared_int4_0.z) return;
    {
    FLOAT4 bias_val = 0.0f;
  {
    FLOAT4 res = out_w0_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
    write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w0_s1);
    // write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w0_s0);
  }
#if 1
    // if (DST_X + 1 < shared_int4_0.x) {
    if (remain >= 2) {
    write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w1_s1);
    write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
  }
#else
    write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w1_s0);
#endif
    if (remain >= 3) {
    write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w2_s1);
    write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w2_s0);
    }
    if (remain >= 4) {
    write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), out_w3_s1);
    write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), out_w3_s0);
    }
    if (remain >= 2) {
    }
    if (remain >= 3) {
    }
    if (remain >= 4) {
    }
    }
#endif
#else
    if (DST_S + 0 >= shared_int4_0.z) return;
  {
    FLOAT4 bias_val = 0.0f;
  {
    FLOAT4 res = out_w0_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLOAT4 res = out_w1_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 2 < shared_int4_0.x) {
    FLOAT4 res = out_w2_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 3 < shared_int4_0.x) {
    FLOAT4 res = out_w3_s0 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  }
  if (DST_S + 1 >= shared_int4_0.z) return;
  {
    FLOAT4 bias_val = 0.0f;
  {
    FLOAT4 res = out_w0_s1 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLOAT4 res = out_w1_s1 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 2 < shared_int4_0.x) {
    FLOAT4 res = out_w2_s1 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 3 < shared_int4_0.x) {
    FLOAT4 res = out_w3_s1 + bias_val;
    {
res = max(res, (FLOAT)(0.0f));
}
write_imageh(output, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  }
#endif
}

__kernel void Conv2DGS3DMulti(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights_c0, __read_only image2d_t weights_c1,
    __read_only image2d_t weights_c2, __read_only image2d_t weights_c3,
    __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_idx         = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_idx);

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);

    const int batch_idx = mul24((output_bh_idx / output_wh.y), input_wh.y);
    const int weights_h_idx = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                              mul24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0), kernel_wh.x);

    FLOAT4 in0, in1, in2, in3;
    FLOAT4 weights0, weights1, weights2, weights3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx = weights_h_idx;
        for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
            int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE(0, input_w_base);
                READ_INPUT_IMAGE(1, input_w_base);
                READ_INPUT_IMAGE(2, input_w_base);
                READ_INPUT_IMAGE(3, input_w_base);

                weights0 = RI_F(weights_c0, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
                weights1 = RI_F(weights_c1, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
                weights2 = RI_F(weights_c2, SAMPLER, (int2)(input_c_block_idx, weights_y_idx));
                weights3 = RI_F(weights_c3, SAMPLER, (int2)(input_c_block_idx, weights_y_idx++));

                CALCULATE_OUTPUT(0);
                CALCULATE_OUTPUT(1);
                CALCULATE_OUTPUT(2);
                CALCULATE_OUTPUT(3);
            }
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, output_w_idx,
                               output_bh_idx, remain);
}

__kernel void Conv2DGS3Dhb2(
    GLOBAL_SIZE_3_DIMS __read_only image2d_t input,
    __read_only image2d_t weights, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int in_channel_block_length, __private const int2 output_wh,
    __private const int2 kernel_wh, __private const int2 stride_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int out_width_blocks, __private const int out_height_blocks) {
    // deal with 2 dim image : dim0 = channel + width | dim1 = batch + height
    const int out_channel_block_idx = get_global_id(0);
    const int out_width_block_idx   = get_global_id(1);
    const int output_bh_block_idx   = get_global_id(2);
    DEAL_NON_UNIFORM_DIM3(out_channel_block_idx, out_width_block_idx, output_bh_block_idx);

    FLOAT4 out0_0 = RI_F(bias, SAMPLER, (int2)(out_channel_block_idx, 0));
    FLOAT4 out0_1 = out0_0;
    FLOAT4 out0_2 = out0_0;
    FLOAT4 out0_3 = out0_0;
    FLOAT4 out1_0 = out0_0;
    FLOAT4 out1_1 = out0_0;
    FLOAT4 out1_2 = out0_0;
    FLOAT4 out1_3 = out0_0;

    int in_width0 = mad24(out_width_block_idx, stride_wh.x << 2, -padding_wh.x);
    int in_width1 = in_width0 + stride_wh.x;
    int in_width2 = in_width0 + stride_wh.x * 2;
    int in_width3 = in_width0 + stride_wh.x * 3;

    int out_h_idx = (output_bh_block_idx % out_height_blocks) << 1;
    const int in_height0 = mad24(out_h_idx, stride_wh.y, -padding_wh.y);
    const int in_height1 = in_height0 + stride_wh.y;
    #if 0
    const int height_start = mad24((output_bh_idx % output_wh.y), stride_wh.y << 1, -padding_wh.y);
    int in_height_start = mad24(select(0, (-height_start + dilation_wh.y - 1) / dilation_wh.y, height_start < 0),
                                dilation_wh.y, height_start);
    int in_height_end = min(mad24(kernel_wh.y, dilation_wh.y, height_start), input_wh.y);
    #endif

    const int batch_idx = mul24((output_bh_block_idx / out_height_blocks), input_wh.y);
    const int weights_h_idx0 = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                               mul24(select(0, (-in_height0 + dilation_wh.y - 1) / dilation_wh.y, in_height0 < 0), kernel_wh.x);
    const int weights_h_idx1 = mul24(out_channel_block_idx, mul24(kernel_wh.x, kernel_wh.y)) +
                               mul24(select(0, (-in_height1 + dilation_wh.y - 1) / dilation_wh.y, in_height1 < 0), kernel_wh.x);

    FLOAT4 in0_0, in0_1, in0_2, in0_3;
    FLOAT4 in1_0, in1_1, in1_2, in1_3;
    FLOAT4 weights0_0, weights0_1, weights0_2, weights0_3;
    FLOAT4 weights1_0, weights1_1, weights1_2, weights1_3;
    for (int input_c_block_idx = 0; input_c_block_idx < in_channel_block_length; ++input_c_block_idx) {
        const int in_idx  = mul24(input_c_block_idx, input_wh.x);
        int weights_x_idx = input_c_block_idx << 2;
        int weights_y_idx0 = weights_h_idx0;
        int weights_y_idx1 = weights_h_idx1;
        // for (int iy = in_height_start; iy < in_height_end; iy += dilation_wh.y) {
        for (int h = 0; h < kernel_wh.y; h++) {
            int input_h_base = mul24(h, dilation_wh.y);
            int in_hb_value0 = in_height0 + input_h_base;
            in_hb_value0 =
                select(in_hb_value0, -1, (in_hb_value0 < 0 || in_hb_value0 >= input_wh.y));
            int in_hb_value1 = in_height1 + input_h_base;
            in_hb_value1 =
                select(in_hb_value1, -1, (in_hb_value1 < 0 || in_hb_value1 >= input_wh.y));
            // int in_hb_value = iy + batch_idx;
            for (int w = 0; w < kernel_wh.x; w++) {
                int input_w_base = mul24(w, dilation_wh.x);
                READ_INPUT_IMAGE_2(0, input_w_base, input_h_base);
                READ_INPUT_IMAGE_2(1, input_w_base, input_h_base);
                READ_INPUT_IMAGE_2(2, input_w_base, input_h_base);
                READ_INPUT_IMAGE_2(3, input_w_base, input_h_base);

                weights0_0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx0));
                weights0_1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx0));
                weights0_2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx0));
                weights0_3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx0++));

                weights1_0 = RI_F(weights, SAMPLER, (int2)(weights_x_idx, weights_y_idx1));
                weights1_1 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 1, weights_y_idx1));
                weights1_2 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 2, weights_y_idx1));
                weights1_3 = RI_F(weights, SAMPLER, (int2)(weights_x_idx + 3, weights_y_idx1++));

                CALCULATE_OUTPUT_2(0);
                CALCULATE_OUTPUT_2(1);
                CALCULATE_OUTPUT_2(2);
                CALCULATE_OUTPUT_2(3);
            }
        }
    }

    out0_0 = ActivationProcess(out0_0);
    out0_1 = ActivationProcess(out0_1);
    out0_2 = ActivationProcess(out0_2);
    out0_3 = ActivationProcess(out0_3);

    out1_0 = ActivationProcess(out1_0);
    out1_1 = ActivationProcess(out1_1);
    out1_2 = ActivationProcess(out1_2);
    out1_3 = ActivationProcess(out1_3);

    const int out_x_base = mul24(out_channel_block_idx, output_wh.x);
    int out_x_idx        = out_width_block_idx << 2;

    const int remain = output_wh.x - out_x_idx;
    int output_w_idx = out_x_base + out_x_idx;
    int output_bh_idx0 = output_bh_block_idx << 1;
    int output_bh_idx1 = output_bh_idx0 + 1;
    WriteOutputAntiOutOfBounds(output, out0_0, out0_1, out0_2, out0_3, output_w_idx,
                               output_bh_idx0, remain);

    if ((out_h_idx + 1) < output_wh.y) {
        WriteOutputAntiOutOfBounds(output, out1_0, out1_1, out1_2, out1_3, output_w_idx,
                                   output_bh_idx1, remain);
    }
}

__kernel void DepthwiseConv2DS1(GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
                                __read_only image2d_t filter,
                                __read_only image2d_t bias,
                                __write_only image2d_t output,
                                __private const int2 input_wh,
                                __private const int2 output_wh,
                                __private const int2 kernel_wh,
                                __private const int2 padding_wh) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightBlockIdx  = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightBlockIdx);
    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int in_width0         = outWidthBlockidx4 - padding_wh.x;
    const int in_width1         = in_width0 + 1;
    const int in_width2         = in_width0 + 2;
    const int in_width3         = in_width0 + 3;

    int heightIdx = outHeightBlockIdx % output_wh.y - padding_wh.y;
    const int outBatchIdx =
        mul24((outHeightBlockIdx / output_wh.y), input_wh.y);
    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);

    const int inWidthIdx0 = select(in_idx + in_width0, -1, (in_width0 < 0 || in_width0 >= input_wh.x));
    const int inWidthIdx1 = select(in_idx + in_width1, -1, (in_width1 < 0 || in_width1 >= input_wh.x));
    const int inWidthIdx2 = select(in_idx + in_width2, -1, (in_width2 < 0 || in_width2 >= input_wh.x));

    FLOAT4 in0, in1, in2, in3;
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx++;
        in1 = RI_F(input, SAMPLER, (int2)(inWidthIdx0, in_hb_value));
        in2 = RI_F(input, SAMPLER, (int2)(inWidthIdx1, in_hb_value));
        in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx2, in_hb_value));
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);

            in0 = in1;
            in1 = in2;
            in2 = in3;

            int inWidthIdx = in_width3 + kw;
            inWidthIdx     = select(in_idx + inWidthIdx, -1, (inWidthIdx < 0 || inWidthIdx >= input_wh.x));
            in3 = RI_F(input, SAMPLER, (int2)(inWidthIdx, in_hb_value));

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int remain = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;
    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightBlockIdx, remain);
}

__kernel void DepthwiseConv2D(
    GLOBAL_SIZE_2_DIMS __read_only image2d_t input,
    __read_only image2d_t filter, __read_only image2d_t bias,
    __write_only image2d_t output, __private const int2 input_wh,
    __private const int2 output_wh, __private const int2 kernel_wh,
    __private const int2 padding_wh, __private const int2 dilation_wh,
    __private const int2 stride_wh) {
    const int outChannelWidthIdx = get_global_id(0);
    const int outHeightIdx       = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(outChannelWidthIdx, outHeightIdx);

    int ow4                      = (output_wh.x + 3) / 4;
    const int outChannelBlockIdx = outChannelWidthIdx / ow4;
    const int outWidthBlockidx   = outChannelWidthIdx % ow4;

    const int inChannelBlockIdx = outChannelBlockIdx;

    FLOAT4 out0 = RI_F(bias, SAMPLER, (int2)(outChannelBlockIdx, 0));
    FLOAT4 out1 = out0;
    FLOAT4 out2 = out0;
    FLOAT4 out3 = out0;

    const int in_width0 = mad24(outWidthBlockidx, stride_wh.x << 2, -padding_wh.x);
    const int in_width1 = in_width0 + stride_wh.x;
    const int in_width2 = in_width1 + stride_wh.x;
    const int in_width3 = in_width2 + stride_wh.x;
    int heightIdx = mad24(outHeightIdx % output_wh.y, stride_wh.y, -padding_wh.y);

    const int outBatchIdx = mul24((outHeightIdx / output_wh.y), input_wh.y);

    const int in_idx = mul24(inChannelBlockIdx, input_wh.x);
    for (int kh = 0; kh < kernel_wh.y; kh++) {
        int in_hb_value = select(heightIdx + outBatchIdx, -1, (heightIdx < 0 || heightIdx >= input_wh.y));
        heightIdx += dilation_wh.y;
        for (int kw = 0; kw < kernel_wh.x; kw++) {
            int filterIdx = mad24(kh, kernel_wh.x, kw);
            FLOAT4 in0, in1, in2, in3;
            int inWidthIdx = mul24(kw, dilation_wh.x);

            READ_INPUT_IMAGE(0, inWidthIdx);
            READ_INPUT_IMAGE(1, inWidthIdx);
            READ_INPUT_IMAGE(2, inWidthIdx);
            READ_INPUT_IMAGE(3, inWidthIdx);

            FLOAT4 weights = RI_F(filter, SAMPLER, (int2)(filterIdx, inChannelBlockIdx));

            out0 = mad(in0, weights, out0);
            out1 = mad(in1, weights, out1);
            out2 = mad(in2, weights, out2);
            out3 = mad(in3, weights, out3);
        }
    }

    out0 = ActivationProcess(out0);
    out1 = ActivationProcess(out1);
    out2 = ActivationProcess(out2);
    out3 = ActivationProcess(out3);

    const int outWidthBlockidx4 = outWidthBlockidx << 2;
    const int remain            = output_wh.x - outWidthBlockidx4;
    int outWidthIdx = mul24(outChannelBlockIdx, output_wh.x) + outWidthBlockidx4;

    WriteOutputAntiOutOfBounds(output, out0, out1, out2, out3, outWidthIdx,
                               outHeightIdx, remain);
}
