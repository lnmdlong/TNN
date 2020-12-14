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

#include "test/unit_test/layer_test/layer_test.h"
#include "test/unit_test/unit_test_common.h"
#include "test/unit_test/utils/network_helpers.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {

#define SPECIAL_DEBUG
class ConvLayerTest : public LayerTest,
                      public ::testing::WithParamInterface<
#ifndef SPECIAL_DEBUG
                          std::tuple<int, int, int, int, int, int, int, int, DataType, ActivationType>> {};
#else
                          std::tuple<int, int, int, int, int, int, int, int, int, int, DataType, ActivationType>> {};
#endif

INSTANTIATE_TEST_SUITE_P(LayerTest, ConvLayerTest,
                         ::testing::Combine(  // batch
#ifndef SPECIAL_DEBUG
                             testing::Values(1),
                             // channel
                             testing::Values(1, 2, 3, 4, 10, 32),
                             // hw
                             testing::Values(9, 10, 16, 19),
                             // group
                             testing::Values(1, 2),
                             // kernel
                             testing::Values(1, 2, 3, 5),
                             // dilation
                             testing::Values(1, 2),
                             // stride
                             testing::Values(1, 2),
                             // pads
                             testing::Values(0, 1),
                             // data_type
                             testing::Values(DATA_TYPE_FLOAT),
                             // activation_type
                             testing::Values(ActivationType_None, ActivationType_ReLU, ActivationType_ReLU6,
                                             ActivationType_SIGMOID_MUL)
#else
                            testing::Values(1),
                             // input channel
                             testing::Values(1280),
                             // output channel
                             testing::Values(1001),
                             // hw
                             testing::Values(1),
                             // group
                             testing::Values(1),
                             // kernel
                             testing::Values(1),
                             // dilation
                             testing::Values(1),
                             // stride
                             testing::Values(1),
                             // pads
                             testing::Values(0),
                             // pad_type
                             testing::Values(0),
                             // data_type
                             testing::Values(DATA_TYPE_FLOAT),
                             // activation_type
                             testing::Values(ActivationType_None)
#endif
                            ));

TEST_P(ConvLayerTest, ConvLayer) {
    // get param
#ifndef SPECIAL_DEBUG
    int batch             = std::get<0>(GetParam());
    int channel_per_group = std::get<1>(GetParam());
    int input_size        = std::get<2>(GetParam());
    int group             = std::get<3>(GetParam());
    int channel           = group * channel_per_group;
    int kernel            = std::get<4>(GetParam());
    int dilation          = std::get<5>(GetParam());
    int stride            = std::get<6>(GetParam());
    int pad               = std::get<7>(GetParam());
    auto dtype            = std::get<8>(GetParam());
    int activation_type   = std::get<9>(GetParam());
#else
    int batch             = std::get<0>(GetParam());
    int input_channel     = std::get<1>(GetParam());
    int output_channel    = std::get<2>(GetParam());
    int input_size        = std::get<3>(GetParam());
    int group             = std::get<4>(GetParam());
    int kernel            = std::get<5>(GetParam());
    int dilation          = std::get<6>(GetParam());
    int stride            = std::get<7>(GetParam());
    int pad               = std::get<8>(GetParam());
    int pad_type          = std::get<9>(GetParam());
    auto dtype            = std::get<10>(GetParam());
    int activation_type   = std::get<11>(GetParam());
#endif
    DeviceType dev        = ConvertDeviceType(FLAGS_dt);

    auto precision = PRECISION_AUTO;
    if (DEVICE_ARM == dev && ActivationType_SIGMOID_MUL) {
        if (DATA_TYPE_FLOAT == dtype) {
            precision = PRECISION_HIGH;
        } else {
            GTEST_SKIP();
        }
    }

#ifndef SPECIAL_DEBUG
    if (((channel_per_group % 4) != 0) && DEVICE_METAL == dev) {
        GTEST_SKIP();
    }
#endif

    if (activation_type != ActivationType_None && DEVICE_HUAWEI_NPU == dev) {
        GTEST_SKIP();
    }

    // param
    std::shared_ptr<ConvLayerParam> param(new ConvLayerParam());
    param->name            = "Conv";
#ifndef SPECIAL_DEBUG
    param->input_channel   = channel;
    param->output_channel  = channel;
#else
    param->input_channel   = input_channel;
    param->output_channel  = output_channel;
#endif
    param->group           = group;
    param->kernels         = {kernel, kernel};
    param->dialations      = {dilation, dilation};
    param->strides         = {stride, stride};
    param->pads            = {pad, pad, pad, pad};
    param->bias            = 1;
    param->activation_type = activation_type;
#ifdef SPECIAL_DEBUG
    param->pad_type        = pad_type;
#endif

    // generate interpreter
#ifndef SPECIAL_DEBUG
    std::vector<int> input_dims = {batch, channel, input_size, input_size};
#else
    std::vector<int> input_dims = {batch, input_channel, input_size, input_size};
#endif
    auto interpreter            = GenerateInterpreter("Convolution", {input_dims}, param);
    Run(interpreter, precision);
}

}  // namespace TNN_NS
