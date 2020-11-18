// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "tnn/device/opencl/acc/convolution/opencl_conv_layer_common_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/string_utils_inner.h"

// #define ENABLE_HEIGHT_BLOCKING
// #define ENABLE_REARRANGE_WEIGHTS
// #define ENABLE_MULTI_WEIGHTS
// #define ENABLE_WEIGHTS_BUFFER
namespace TNN_NS {

bool OpenCLConvLayerCommonAcc::IsPrefered(const ConvLayerParam *param, const std::vector<Blob *> &,
                                          const std::vector<Blob *> &) {
    if (!param) {
        return false;
    }

    return true;
}

Status OpenCLConvLayerCommonAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                      const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Conv Common Acc\n");
    LOGE("dlmeng: Init Conv Common Acc\n");

    Status ret = OpenCLConvLayerAccImpl::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    conv_type_ = CT_CONV_COMMON;
    op_name_   = "Conv_" + ToString(conv_params_.kernel_x) + "x" + ToString(conv_params_.kernel_y);

    if(conv_params_.kernel_x != conv_params_.kernel_y) {
        run_3d_ndrange_ = false;
    }

    #ifdef ENABLE_WEIGHTS_BUFFER
    use_buffer_ = true;
    #endif
    LOGE("dlmeng: Allocate Weights Bias start\n");
    ret = AllocateWeightsBias(resource);
    CHECK_TNN_OK(ret)
    LOGE("dlmeng: Allocate Weights Bias succeed\n");

    // create kernel
    std::set<std::string> build_options;
    if (conv_params_.activation_type == ActivationType_ReLU) {
        build_options.emplace("-DRELU");
    } else if (conv_params_.activation_type == ActivationType_ReLU6) {
        build_options.emplace("-DRELU6");
    }

    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = output_dims[2];
    const int output_width  = output_dims[3];
    const int output_channel = output_dims[1];

    const int input_height   = input_dims[2];
    const int input_width    = input_dims[3];
    const int input_channel  = input_dims[1];

    run_3d_ndrange_ = true;
    is_channel_blocking_ = false;
    bool is_conv_3x3s1p1d1 = (conv_params_.kernel_x == 3 && conv_params_.kernel_y == 3 &&
                              conv_params_.stride_x == 1 && conv_params_.stride_y == 1 &&
                              conv_params_.dilation_x == 1 && conv_params_.dilation_y == 1);
    std::string kernel_name = "Conv2D";
    if (run_3d_ndrange_) {
#ifndef ENABLE_HEIGHT_BLOCKING
        kernel_name = "Conv2DGS3D";
#else
        kernel_name = "Conv2DGS3Dhb2";
#endif
        if (is_conv_3x3s1p1d1) {
#ifdef ENABLE_REARRANGE_WEIGHTS
            kernel_name = "Conv2DGS3D3x3s1d1Rearrange";
#else
            kernel_name = "Conv2DGS3D3x3s1d1";
            // kernel_name = "Conv2DGS3D3x3s1d1LoopRearrange";
#endif
#ifdef ENABLE_MULTI_WEIGHTS
            // kernel_name = "Conv2DGS3DMulti";
#endif
            if (output_channel > 4) {
                is_channel_blocking_ = true;
                kernel_name = "Conv2DGS3Dcb2";
                // kernel_name = "Conv2DGS3D3x3s1d1cb2";
            }
        }
    }
    else if (is_conv_3x3s1p1d1) {
        kernel_name = "Conv2D3x3s1d1";
    }

    if (use_buffer_) {
        kernel_name += "_MIX";
    }

#if 1
    LOGE("dlmeng: kernel_name: %s, kernel: [%d, %d], stride: [%d, %d], input: [%d, %d, %d], out: [%d, %d, %d]\n",
         kernel_name.c_str(),
         conv_params_.kernel_x, conv_params_.kernel_y, conv_params_.stride_x,
         conv_params_.stride_y,
         input_channel, input_height, input_width,
         output_channel, output_height, output_width);
#endif

#ifndef ENABLE_CONV_EXP
    ret = CreateExecuteUnit(execute_units_[0], "convolution", kernel_name, build_options);
#else
    // std::string exp_kernel_name = "main_function";
    std::string exp_kernel_name = "main_function_w4h1s2";
    LOGE("dlmeng: create exp function: %s\n", exp_kernel_name.c_str());
    ret = CreateExecuteUnit(execute_units_[0], "convolution_exp", exp_kernel_name, build_options);
#endif
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLConvLayerCommonAcc::~OpenCLConvLayerCommonAcc() {}

Status OpenCLConvLayerCommonAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv Common Acc Reshape\n");
    auto input_dims  = inputs[0]->GetBlobDesc().dims;
    auto output_dims = outputs[0]->GetBlobDesc().dims;

    const int output_height = output_dims[2];
    const int output_width  = output_dims[3];

    const int input_height   = input_dims[2];
    const int input_width    = input_dims[3];

    int input_imageshape[2]  = {input_width, input_height};
    int output_imageshape[2] = {output_width, output_height};
    int kernel_shape[2]      = {conv_params_.kernel_x, conv_params_.kernel_y};
    int stride_shape[2]      = {conv_params_.stride_x, conv_params_.stride_y};
    int padding_shape[2]     = {conv_params_.pad_x, conv_params_.pad_y};
    int dilation_shape[2]    = {conv_params_.dilation_x, conv_params_.dilation_y};

#ifndef ENABLE_CONV_EXP
    if (run_3d_ndrange_) {
        if (is_channel_blocking_) {
            #ifndef TNN_HEIGHT_DEBUG
            execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 8)),
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
            #else
            #if 0
            execute_units_[0].global_work_size = {
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2]),
                                            static_cast<uint32_t>(UP_DIV(output_dims[1], 8))};
            #else
            execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 8)),
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
            #endif
            #endif
        } else {
            #ifndef TNN_HEIGHT_DEBUG
            execute_units_[0].global_work_size = {static_cast<uint32_t>(UP_DIV(output_dims[1], 4)),
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            #ifndef ENABLE_HEIGHT_BLOCKING
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
                                            #else
                                            static_cast<uint32_t>(output_dims[0] * UP_DIV(output_dims[2], 2))};
                                            #endif
            #else
            execute_units_[0].global_work_size = {
                                            static_cast<uint32_t>(UP_DIV(output_dims[3], 4)),
                                            static_cast<uint32_t>(output_dims[0] * output_dims[2]),
                                            #ifndef ENABLE_HEIGHT_BLOCKING
                                            static_cast<uint32_t>(UP_DIV(output_dims[1], 4))};
                                            #else
                                            static_cast<uint32_t>(output_dims[0] * UP_DIV(output_dims[2], 2))};
                                            #endif
            #endif
        }
        if(kernel_shape[0] == 3 && kernel_shape[1] == 3) {
            execute_units_[0].local_work_size  = Conv2dCommonLocalWS3DKernel3x3(
                execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1], execute_units_[0].workgroupsize_max);
#if 1
            auto gws = execute_units_[0].global_work_size;
            if (gws[0] == 18 && gws[1] == 71 && gws[2] == 24) {
                execute_units_[0].local_work_size = {10, 1, 24};
            }
#endif
        } else {
            execute_units_[0].local_work_size  = Conv2dCommonLocalWS3DGeneral(
                execute_units_[0].global_work_size, kernel_shape[0] * kernel_shape[1], execute_units_[0].workgroupsize_max);
        }
    } else {
        execute_units_[0].global_work_size = {
            static_cast<uint32_t>(UP_DIV(output_dims[1], 4) * UP_DIV(output_dims[3], 4)),
            static_cast<uint32_t>(output_dims[0] * output_dims[2])};
        execute_units_[0].local_work_size = LocalWS2DDefault(execute_units_[0]);
#if 0
        auto gws = execute_units_[0].global_work_size;
        if (gws[0] == 864 && gws[1] == 71) {
            execute_units_[0].local_work_size = {8, 2};
        }
#endif
    }

#if 1
    LOGE("dlmeng: gws: [%d, %d, %d]\n", execute_units_[0].global_work_size[0], execute_units_[0].global_work_size[1],
         execute_units_[0].global_work_size[2]);
#endif


    const int input_channels = input_dims[1];
    const int input_channel_blocks = UP_DIV(input_channels, 4);

    const int output_channels = output_dims[1];
    const int output_channel_blocks = UP_DIV(output_channels, 4);

    uint32_t idx = 0;
    for (auto gws : execute_units_[0].global_work_size) {
        execute_units_[0].ocl_kernel.setArg(idx++, gws);
    }

#ifndef TNN_HEIGHT_DEBUG
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
#else
#if 0
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)conv_exp_src_->GetData()));
#else
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
#endif
#endif
#ifndef ENABLE_MULTI_WEIGHTS
    if (!use_buffer_) {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    } else {
        LOGE("dlmeng: set weights buffer args\n");
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)ocl_weights_->GetData()));
    }
#else
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights0_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights1_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights2_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights3_->GetData()));
#endif

    if (!use_buffer_) {
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    } else {
        LOGE("dlmeng: set bias buffer args\n");
        execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)ocl_bias_->GetData()));
    }
#ifndef TNN_HEIGHT_DEBUG
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
#else
#if 0
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)conv_exp_dst_->GetData()));
#else
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
#endif
#endif
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(input_imageshape), input_imageshape);
    
    execute_units_[0].ocl_kernel.setArg(idx++, input_channel_blocks);
    if (is_channel_blocking_) {
        execute_units_[0].ocl_kernel.setArg(idx++, output_channel_blocks);
    }
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(output_imageshape), output_imageshape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(kernel_shape), kernel_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(stride_shape), stride_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(padding_shape), padding_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(dilation_shape), dilation_shape);
    execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(output_width, 4));
#ifdef ENABLE_HEIGHT_BLOCKING
    execute_units_[0].ocl_kernel.setArg(idx++, UP_DIV(output_height, 2));
#endif
#ifdef ENABLE_REARRANGE_WEIGHTS
    execute_units_[0].ocl_kernel.setArg(idx++, input_channels);
#endif
#if 1
    int shared_int4_0[4] = {output_width, output_height, output_channel_blocks, conv_params_.stride_x};
    int shared_int4_1[4] = {-conv_params_.pad_x, conv_params_.stride_y, -conv_params_.pad_y, conv_params_.kernel_x};
    int shared_int4_2[4] = {conv_params_.dilation_x, conv_params_.kernel_y, conv_params_.dilation_y, input_channel_blocks};
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_0), shared_int4_0);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_1), shared_int4_1);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_2), shared_int4_2);
#endif
#else
    #if 0
    execute_units_[0].global_work_size = {40, 36, 24};
    execute_units_[0].local_work_size = {10, 1, 24};
    #elif 1
    execute_units_[0].global_work_size = {20, 71, 24};
    execute_units_[0].local_work_size = {10, 1, 24};
    #endif
    
    // exp args
    uint32_t idx = 0;
    const int input_channels = input_dims[1];
    const int input_channel_blocks = UP_DIV(input_channels, 4);
    const int output_channels = output_dims[1];
    const int output_channel_blocks = UP_DIV(output_channels, 4);
    int shared_int4_0[4] = {output_width, output_height, output_channel_blocks, conv_params_.stride_x};
    int shared_int4_1[4] = {-conv_params_.pad_x, conv_params_.stride_y, -conv_params_.pad_y, conv_params_.kernel_x};
    int shared_int4_2[4] = {conv_params_.dilation_x, conv_params_.kernel_y, conv_params_.dilation_y, input_channel_blocks};
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Buffer *)ocl_bias_buffer_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)conv_exp_dst_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)conv_exp_src_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights0_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights1_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights2_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights3_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_0), shared_int4_0);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_1), shared_int4_1);
    execute_units_[0].ocl_kernel.setArg(idx++, sizeof(shared_int4_2), shared_int4_2);
#endif

    return TNN_OK;
}

}  // namespace TNN_NS
