/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <iostream>
#include <numeric>
#include <math.h>
#include <string>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ittnotify.h>
#include "mkldnn.hpp"
#include "test_convolution_forward_common.hpp"

using namespace mkldnn;

inline uint64_t rdtscp(void) {
    uint64_t rax,rdx;
    uint32_t aux;

    __asm__ volatile( "rdtscp\n" : "=a" (rax), "=d" (rdx), "=c" (aux) : : );

    return (rdx << 32) + rax;
}

inline int right_padding_(int i, int o, int k, int p, int s, int d = 0) {
    return (o - 1) * s + (k - 1) * (d + 1) - (p + i - 1);
}

void simple_net(int sparsity, int n, int mb, int ic, int ih, int iw, int oc, int strh,
        int strw, int kh, int kw)
{

    int padh = kh / 2, padw = kw / 2;
    int oh = ih / strh, ow = iw / strw;
    int dilh = 0, dilw = 0;



    std::cout << "mb=" << mb << " ic=" << ic << " ih=" << ih << " iw=" << iw
        << " oc=" << oc << " oh=" << oh << " ow=" << ow << " kh=" << kh << " kw=" << kw
        << " strh=" << strh << " strw=" << strw << std::endl;


    float *src_data = (float *) aligned_alloc(64, mb * ic * ih * iw * sizeof(float));
    float *dst_data = (float *) aligned_alloc(64, mb * oc * oh * ow * sizeof(float));
    float *wei_data = (float *) aligned_alloc(64, oc * ic * kh * kw * sizeof(float));
    float *bias_data = (float *) aligned_alloc(64, oc * sizeof(float));

    float *wei_ref_data = (float *) aligned_alloc(64, oc * ic * kh * kw * sizeof(float));

    for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        dst_data[i] = i;
    }

    for (size_t i = 0; i < mb * ic * ih * iw; ++i) {
        if (rand() % 100 >= sparsity) {
            src_data[i] = i;
        } else {
            src_data[i] = 0.0;
        }
    }

    auto cpu_engine = engine(engine::cpu, 0);
    auto strm = stream(cpu_engine);

    auto c_src_desc = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format_tag::nChw16c);
    auto c_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format_tag::OIhw16i16o);
    auto c_dst_desc = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format_tag::nChw16c);
    auto c_bias_desc = memory::desc({ oc }, memory::data_type::f32,
                                        memory::format_tag::x);

    auto c_src_desc_f = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format_tag::nChw16c);
    auto c_dst_desc_f = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format_tag::nChw16c);

    auto c_src = memory(c_src_desc, cpu_engine, src_data);
    auto c_diff_weights = memory(c_weights_desc, cpu_engine, wei_data);
    auto c_diff_dst = memory(c_dst_desc, cpu_engine, dst_data);
    auto c_diff_bias = memory(c_bias_desc, cpu_engine, bias_data);

    auto c_ref_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format_tag::OIhw16i16o);
    auto c_diff_ref_weights = memory(c_ref_weights_desc, cpu_engine, wei_ref_data);

    memory::dims padR = {
        right_padding_(ih, oh, kh, padh, strh, dilh),
        right_padding_(iw, ow, kw, padw, strw, dilw)
    };

    auto conv_desc = convolution_forward::desc(
            prop_kind::forward_training, convolution_direct, c_src_desc_f,
            c_weights_desc, c_dst_desc_f,
            { strh, strw }, { dilh, dilw },
            { padh, padw }, padR, padding_kind::zero);
    auto conv_primitive_desc = convolution_forward::primitive_desc(
            conv_desc, cpu_engine);

    auto conv_bwd_weights_desc = convolution_backward_weights::desc(
            convolution_direct, c_src_desc, c_weights_desc,
            c_dst_desc,
            { strh, strw }, { dilh, dilw },
            { padh, padw }, padR, padding_kind::zero);
    auto conv_bwd_weights_primitive_desc
        = convolution_backward_weights::primitive_desc(
                conv_bwd_weights_desc, cpu_engine, conv_primitive_desc);

    uint64_t min_time = UINT64_MAX;
    
    __itt_resume();
    for (int i = 0; i < n; ++i) {
        std::cout << "iter " << i << std::endl;

        uint64_t start = rdtscp();

        convolution_backward_weights(conv_bwd_weights_primitive_desc).execute(
                strm, {
                {MKLDNN_ARG_DIFF_DST, c_diff_dst},
                {MKLDNN_ARG_SRC, c_src},
                {MKLDNN_ARG_DIFF_WEIGHTS, c_diff_weights}});

        uint64_t end = rdtscp();

        uint64_t time = end - start;
        if (time < min_time) {
            min_time = time;
        }
        std::cout << "min time: " << min_time << std::endl;
    }
    __itt_pause();

}

int main(int argc, char **argv)
{
    int sparsity = 50;
    int n = 1;

    int mb = 32;
    int ic = 512, ih = 28, iw = 28;
    int oc = 512;
    int kh = 3, kw = 3;

    int strh = 1, strw = 1;

    if (argc > 1) {
        sparsity = atoi(argv[1]);
    }

    if (argc > 2) {
        n = atoi(argv[2]);
    }

    if (argc > 3) {
        mb = atoi(argv[3]);
    }

    if (argc > 4) {
        ic = atoi(argv[4]);
    }

    if (argc > 5) {
        ih = atoi(argv[5]);
    }

    if (argc > 6) {
        iw = atoi(argv[6]);
    }

    if (argc > 7) {
        oc = atoi(argv[7]);
    }

    if (argc > 8) {
        strh = atoi(argv[8]);
    }

    if (argc > 9) {
        strw = atoi(argv[9]);
    }

    if (argc > 10) {
        kh = atoi(argv[10]);
    }

    if (argc > 11) {
        kw = atoi(argv[11]);
    }

    try
    {
        simple_net(sparsity, n, mb, ic, ih, iw, oc, strh, strw, kh, kw);
        std::cout << "passed" << std::endl;
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
