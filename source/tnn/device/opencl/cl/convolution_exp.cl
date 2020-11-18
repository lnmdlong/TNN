__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define ACCUM_FLT4 half4
#define FLT half
#define FLT2 half2
#define FLT3 half3
#define FLT4 half4
#define TO_FLT4 convert_half4
#define TO_ACCUM_TYPE convert_half4
#define TO_ACCUM_FLT convert_half
__kernel void main_function(
__global half4* biases_buffer,
  __write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  __read_only image2d_t weights0_tex2d,
  __read_only image2d_t weights1_tex2d,
  __read_only image2d_t weights2_tex2d,
  __read_only image2d_t weights3_tex2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {
  int DST_X = get_global_id(0) * 2;
  int DST_Y = get_global_id(1) * 2;
  int DST_S = get_global_id(2) * 2;
  if (DST_X >= shared_int4_0.x || DST_Y >= shared_int4_0.y || DST_S >= shared_int4_0.z) {
    return;
  }
  ACCUM_FLT4 r_w0_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w0_h1_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h1_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w0_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w0_h1_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h1_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int xc0 = (DST_X + 0) * shared_int4_0.w + shared_int4_1.x;
  int xc1 = (DST_X + 1) * shared_int4_0.w + shared_int4_1.x;
  int yc0 = (DST_Y + 0) * shared_int4_1.y + shared_int4_1.z;
  int yc1 = (DST_Y + 1) * shared_int4_1.y + shared_int4_1.z;
  int filter_offset = 0;
    int weights_y_idx_s0 = DST_S * mul24(shared_int4_1.w, shared_int4_2.y);
    int weights_y_idx_s1 = weights_y_idx_s0 + mul24(shared_int4_1.w, shared_int4_2.y);
    int input_width = 73;
  for (int ky = 0; ky < shared_int4_1.w; ++ky) {
  int yck0 = ky * shared_int4_2.x + yc0;
  int yck1 = ky * shared_int4_2.x + yc1;
  for (int kx = 0; kx < shared_int4_2.y; ++kx) {
  int xck0 = kx * shared_int4_2.z + xc0;
  int xck1 = kx * shared_int4_2.z + xc1;
  int s = 0;
  do {
    half4 src_w0_h0;
    half4 src_w1_h0;
    half4 src_w0_h1;
    half4 src_w1_h1;
    #if 1
    FLT4 f0 = read_imageh(weights0_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f1 = read_imageh(weights1_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f2 = read_imageh(weights2_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f3 = read_imageh(weights3_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f4 = read_imageh(weights0_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f5 = read_imageh(weights1_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f6 = read_imageh(weights2_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f7 = read_imageh(weights3_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    #elif 1
    int weights_x_idx = s << 2;
    FLT4 f0 = read_imageh(weights0_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s0));
    FLT4 f1 = read_imageh(weights1_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s0));
    FLT4 f2 = read_imageh(weights2_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s0));
    FLT4 f3 = read_imageh(weights3_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s0));
    FLT4 f4 = read_imageh(weights0_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s1));
    FLT4 f5 = read_imageh(weights1_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s1));
    FLT4 f6 = read_imageh(weights2_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s1));
    FLT4 f7 = read_imageh(weights3_tex2d, smp_none, (int2)(weights_x_idx, weights_y_idx_s1));
    #elif 1
    FLT4 f0 = read_imageh(weights0_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f1 = f0;
    FLT4 f2 = f0;
    FLT4 f3 = f0;
    FLT4 f4 = f0;
    FLT4 f5 = f0;
    FLT4 f6 = f0;
    FLT4 f7 = f0;
    #endif
    filter_offset++;
    #if 1
    src_w0_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck0), (yck0) * shared_int4_2.w + (s)));
    src_w1_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck1), (yck0) * shared_int4_2.w + (s)));
    src_w0_h1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck0), (yck1) * shared_int4_2.w + (s)));
    src_w1_h1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck1), (yck1) * shared_int4_2.w + (s)));
    #elif 1
    src_w0_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((s * input_width + xck0), yck0));
    src_w1_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((s * input_width + xck1), yck0));
    src_w0_h1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((s * input_width + xck0), yck1));
    src_w1_h1 = read_imageh(src_tensor_image2d, smp_zero, (int2)((s * input_width + xck1), yck1));
    #elif 1
    src_w0_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck0), (yck0) * shared_int4_2.w + (s)));
    src_w1_h0 = src_w0_h0;
    src_w0_h1 = src_w0_h0;
    src_w1_h1 = src_w0_h0;
    #endif
    s += 1;
    #if 1
    r_w0_h0_s0 += f0 * src_w0_h0.x;
    r_w1_h0_s0 += f0 * src_w1_h0.x;
    r_w0_h1_s0 += f0 * src_w0_h1.x;
    r_w1_h1_s0 += f0 * src_w1_h1.x;
    r_w0_h0_s0 += f1 * src_w0_h0.y;
    r_w1_h0_s0 += f1 * src_w1_h0.y;
    r_w0_h1_s0 += f1 * src_w0_h1.y;
    r_w1_h1_s0 += f1 * src_w1_h1.y;
    r_w0_h0_s0 += f2 * src_w0_h0.z;
    r_w1_h0_s0 += f2 * src_w1_h0.z;
    r_w0_h1_s0 += f2 * src_w0_h1.z;
    r_w1_h1_s0 += f2 * src_w1_h1.z;
    r_w0_h0_s0 += f3 * src_w0_h0.w;
    r_w1_h0_s0 += f3 * src_w1_h0.w;
    r_w0_h1_s0 += f3 * src_w0_h1.w;
    r_w1_h1_s0 += f3 * src_w1_h1.w;
    r_w0_h0_s1 += f4 * src_w0_h0.x;
    r_w1_h0_s1 += f4 * src_w1_h0.x;
    r_w0_h1_s1 += f4 * src_w0_h1.x;
    r_w1_h1_s1 += f4 * src_w1_h1.x;
    r_w0_h0_s1 += f5 * src_w0_h0.y;
    r_w1_h0_s1 += f5 * src_w1_h0.y;
    r_w0_h1_s1 += f5 * src_w0_h1.y;
    r_w1_h1_s1 += f5 * src_w1_h1.y;
    r_w0_h0_s1 += f6 * src_w0_h0.z;
    r_w1_h0_s1 += f6 * src_w1_h0.z;
    r_w0_h1_s1 += f6 * src_w0_h1.z;
    r_w1_h1_s1 += f6 * src_w1_h1.z;
    r_w0_h0_s1 += f7 * src_w0_h0.w;
    r_w1_h0_s1 += f7 * src_w1_h0.w;
    r_w0_h1_s1 += f7 * src_w0_h1.w;
    r_w1_h1_s1 += f7 * src_w1_h1.w;
    #elif 1
    r_w0_h0_s0 = mad(f0, src_w0_h0.x, r_w0_h0_s0);
    r_w1_h0_s0 = mad(f0, src_w1_h0.x, r_w1_h0_s0);
    r_w0_h1_s0 = mad(f0, src_w0_h1.x, r_w0_h1_s0);
    r_w1_h1_s0 = mad(f0, src_w1_h1.x, r_w1_h1_s0);
    r_w0_h0_s0 = mad(f1, src_w0_h0.y, r_w0_h0_s0);
    r_w1_h0_s0 = mad(f1, src_w1_h0.y, r_w1_h0_s0);
    r_w0_h1_s0 = mad(f1, src_w0_h1.y, r_w0_h1_s0);
    r_w1_h1_s0 = mad(f1, src_w1_h1.y, r_w1_h1_s0);
    r_w0_h0_s0 = mad(f2, src_w0_h0.z, r_w0_h0_s0);
    r_w1_h0_s0 = mad(f2, src_w1_h0.z, r_w1_h0_s0);
    r_w0_h1_s0 = mad(f2, src_w0_h1.z, r_w0_h1_s0);
    r_w1_h1_s0 = mad(f2, src_w1_h1.z, r_w1_h1_s0);
    r_w0_h0_s0 = mad(f3, src_w0_h0.w, r_w0_h0_s0);
    r_w1_h0_s0 = mad(f3, src_w1_h0.w, r_w1_h0_s0);
    r_w0_h1_s0 = mad(f3, src_w0_h1.w, r_w0_h1_s0);
    r_w1_h1_s0 = mad(f3, src_w1_h1.w, r_w1_h1_s0);
    r_w0_h0_s1 = mad(f4, src_w0_h0.x, r_w0_h0_s1);
    r_w1_h0_s1 = mad(f4, src_w1_h0.x, r_w1_h0_s1);
    r_w0_h1_s1 = mad(f4, src_w0_h1.x, r_w0_h1_s1);
    r_w1_h1_s1 = mad(f4, src_w1_h1.x, r_w1_h1_s1);
    r_w0_h0_s1 = mad(f5, src_w0_h0.y, r_w0_h0_s1);
    r_w1_h0_s1 = mad(f5, src_w1_h0.y, r_w1_h0_s1);
    r_w0_h1_s1 = mad(f5, src_w0_h1.y, r_w0_h1_s1);
    r_w1_h1_s1 = mad(f5, src_w1_h1.y, r_w1_h1_s1);
    r_w0_h0_s1 = mad(f6, src_w0_h0.z, r_w0_h0_s1);
    r_w1_h0_s1 = mad(f6, src_w1_h0.z, r_w1_h0_s1);
    r_w0_h1_s1 = mad(f6, src_w0_h1.z, r_w0_h1_s1);
    r_w1_h1_s1 = mad(f6, src_w1_h1.z, r_w1_h1_s1);
    r_w0_h0_s1 = mad(f7, src_w0_h0.w, r_w0_h0_s1);
    r_w1_h0_s1 = mad(f7, src_w1_h0.w, r_w1_h0_s1);
    r_w0_h1_s1 = mad(f7, src_w0_h1.w, r_w0_h1_s1);
    r_w1_h1_s1 = mad(f7, src_w1_h1.w, r_w1_h1_s1);
    #elif 1
    r_w0_h0_s0 += f0 * src_w0_h0.x;
    r_w1_h0_s0 += f1 * src_w1_h0.y;
    r_w0_h1_s0 += f2 * src_w0_h1.z;
    r_w1_h1_s0 += f3 * src_w1_h1.w;
    r_w0_h0_s1 += f4 * src_w0_h0.x;
    r_w1_h0_s1 += f5 * src_w1_h0.y;
    r_w0_h1_s1 += f6 * src_w0_h1.z;
    r_w1_h1_s1 += f7 * src_w1_h1.w;
    #elif 0
    r_w0_h0_s0 += f0 * src_w0_h0;
    r_w1_h0_s0 += f1 * src_w1_h0;
    r_w0_h1_s0 += f2 * src_w0_h1;
    r_w1_h1_s0 += f3 * src_w1_h1;
    r_w0_h0_s1 += f4;
    r_w1_h0_s1 += f5;
    r_w0_h1_s1 += f6;
    r_w1_h1_s1 += f7;
    #endif
  } while (s < shared_int4_2.w);
  weights_y_idx_s0++;
  weights_y_idx_s1++;
  };
  };
  if (DST_S + 0 >= shared_int4_0.z) return;
  {
    FLT4 bias_val = biases_buffer[DST_S + 0];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w1_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_Y + 1 < shared_int4_0.y) {
    FLT4 res = TO_FLT4(r_w0_h1_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 1) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x && DST_Y + 1 < shared_int4_0.y) {
    FLT4 res = TO_FLT4(r_w1_h1_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 1) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  }
  if (DST_S + 1 >= shared_int4_0.z) return;
  {
    FLT4 bias_val = biases_buffer[DST_S + 1];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w1_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_Y + 1 < shared_int4_0.y) {
    FLT4 res = TO_FLT4(r_w0_h1_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 1) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x && DST_Y + 1 < shared_int4_0.y) {
    FLT4 res = TO_FLT4(r_w1_h1_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 1) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  }
}

__kernel void main_function_w4h1s2(
__global half4* biases_buffer,
  __write_only image2d_t dst_tensor_image2d,
  __read_only image2d_t src_tensor_image2d,
  __read_only image2d_t weights0_tex2d,
  __read_only image2d_t weights1_tex2d,
  __read_only image2d_t weights2_tex2d,
  __read_only image2d_t weights3_tex2d,
  int4 shared_int4_0,
  int4 shared_int4_1,
  int4 shared_int4_2) {
  int DST_X = get_global_id(0) * 4;
  int DST_Y = get_global_id(1) * 1;
  int DST_S = get_global_id(2) * 2;
  if (DST_X >= shared_int4_0.x || DST_Y >= shared_int4_0.y || DST_S >= shared_int4_0.z) {
    return;
  }
  ACCUM_FLT4 r_w0_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w2_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w3_h0_s0 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w0_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w1_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w2_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  ACCUM_FLT4 r_w3_h0_s1 = (ACCUM_FLT4)(0.0f, 0.0f, 0.0f, 0.0f);
  int xc0 = (DST_X + 0) * shared_int4_0.w + shared_int4_1.x;
  int xc1 = (DST_X + 1) * shared_int4_0.w + shared_int4_1.x;
  int xc2 = (DST_X + 2) * shared_int4_0.w + shared_int4_1.x;
  int xc3 = (DST_X + 3) * shared_int4_0.w + shared_int4_1.x;
  int yc0 = (DST_Y + 0) * shared_int4_1.y + shared_int4_1.z;
  int filter_offset = 0;
  for (int ky = 0; ky < shared_int4_1.w; ++ky) {
  int yck0 = ky * shared_int4_2.x + yc0;
  for (int kx = 0; kx < shared_int4_2.y; ++kx) {
  int xck0 = kx * shared_int4_2.z + xc0;
  int xck1 = kx * shared_int4_2.z + xc1;
  int xck2 = kx * shared_int4_2.z + xc2;
  int xck3 = kx * shared_int4_2.z + xc3;
  int s = 0;
  do {
    half4 src_w0_h0;
    half4 src_w1_h0;
    half4 src_w2_h0;
    half4 src_w3_h0;
    FLT4 f0 = read_imageh(weights0_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f1 = read_imageh(weights1_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f2 = read_imageh(weights2_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f3 = read_imageh(weights3_tex2d, smp_none, (int2)(DST_S + 0, filter_offset));
    FLT4 f4 = read_imageh(weights0_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f5 = read_imageh(weights1_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f6 = read_imageh(weights2_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    FLT4 f7 = read_imageh(weights3_tex2d, smp_none, (int2)(DST_S + 1, filter_offset));
    filter_offset++;
    src_w0_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck0), (yck0) * shared_int4_2.w + (s)));
    src_w1_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck1), (yck0) * shared_int4_2.w + (s)));
    src_w2_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck2), (yck0) * shared_int4_2.w + (s)));
    src_w3_h0 = read_imageh(src_tensor_image2d, smp_zero, (int2)((xck3), (yck0) * shared_int4_2.w + (s)));
    s += 1;
    r_w0_h0_s0 += f0 * src_w0_h0.x;
    r_w1_h0_s0 += f0 * src_w1_h0.x;
    r_w2_h0_s0 += f0 * src_w2_h0.x;
    r_w3_h0_s0 += f0 * src_w3_h0.x;
    r_w0_h0_s0 += f1 * src_w0_h0.y;
    r_w1_h0_s0 += f1 * src_w1_h0.y;
    r_w2_h0_s0 += f1 * src_w2_h0.y;
    r_w3_h0_s0 += f1 * src_w3_h0.y;
    r_w0_h0_s0 += f2 * src_w0_h0.z;
    r_w1_h0_s0 += f2 * src_w1_h0.z;
    r_w2_h0_s0 += f2 * src_w2_h0.z;
    r_w3_h0_s0 += f2 * src_w3_h0.z;
    r_w0_h0_s0 += f3 * src_w0_h0.w;
    r_w1_h0_s0 += f3 * src_w1_h0.w;
    r_w2_h0_s0 += f3 * src_w2_h0.w;
    r_w3_h0_s0 += f3 * src_w3_h0.w;
    r_w0_h0_s1 += f4 * src_w0_h0.x;
    r_w1_h0_s1 += f4 * src_w1_h0.x;
    r_w2_h0_s1 += f4 * src_w2_h0.x;
    r_w3_h0_s1 += f4 * src_w3_h0.x;
    r_w0_h0_s1 += f5 * src_w0_h0.y;
    r_w1_h0_s1 += f5 * src_w1_h0.y;
    r_w2_h0_s1 += f5 * src_w2_h0.y;
    r_w3_h0_s1 += f5 * src_w3_h0.y;
    r_w0_h0_s1 += f6 * src_w0_h0.z;
    r_w1_h0_s1 += f6 * src_w1_h0.z;
    r_w2_h0_s1 += f6 * src_w2_h0.z;
    r_w3_h0_s1 += f6 * src_w3_h0.z;
    r_w0_h0_s1 += f7 * src_w0_h0.w;
    r_w1_h0_s1 += f7 * src_w1_h0.w;
    r_w2_h0_s1 += f7 * src_w2_h0.w;
    r_w3_h0_s1 += f7 * src_w3_h0.w;
  } while (s < shared_int4_2.w);
  };
  };
  if (DST_S + 0 >= shared_int4_0.z) return;
  {
    FLT4 bias_val = biases_buffer[DST_S + 0];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w1_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 2 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w2_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  if (DST_X + 3 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w3_h0_s0) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 0)), res);
;
  }
  }
  if (DST_S + 1 >= shared_int4_0.z) return;
  {
    FLT4 bias_val = biases_buffer[DST_S + 1];
  {
    FLT4 res = TO_FLT4(r_w0_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 0), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 1 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w1_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 1), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 2 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w2_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 2), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  if (DST_X + 3 < shared_int4_0.x) {
    FLT4 res = TO_FLT4(r_w3_h0_s1) + bias_val;
    {
res = max(res, (FLT)(0.0f));
}
write_imageh(dst_tensor_image2d, (int2)((DST_X + 3), (DST_Y + 0) * shared_int4_0.z + (DST_S + 1)), res);
;
  }
  }
}