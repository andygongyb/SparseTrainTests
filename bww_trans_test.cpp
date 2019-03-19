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

#ifndef VERIFY
#define VERIFY (true)
#endif

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

void compute_ref_conv_bwd_weights(const test_convolution_sizes_t &c,
        const memory &src, const memory &diff_dst, const memory &diff_weights)
{
    float *src_data = (float *)src.get_data_handle();
    float *diff_weights_data
        = (float *)diff_weights.get_data_handle();
    float *diff_dst_data
        = (float *)diff_dst.get_data_handle();

    const memory::desc src_d = src.get_primitive_desc().desc();
    const memory::desc weights_d = diff_weights.get_primitive_desc().desc();
    const memory::desc dst_d = diff_dst.get_primitive_desc().desc();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.ng, c.oc / c.ng, c.ic / c.ng, c.kh, c.kw,
        [&](int g, int oc, int ic, int kh, int kw) {
        size_t widx = g * padded_oc / c.ng * padded_ic / c.ng * c.kh * c.kw
                + oc * padded_ic / c.ng * c.kh * c.kw
                + ic * c.kh * c.kw + kh * c.kw + kw;
        diff_weights_data[map_index(weights_d, widx)] = 0.0;
        for (int mb = 0; mb < c.mb; ++mb) {
            for (int oh = 0; oh < c.oh; ++oh) {
                for (int ow = 0; ow < c.ow; ++ow) {
                    if (ow*c.strw + kw * (1 + c.dilw) < c.padw ||
                        oh*c.strh + kh * (1 + c.dilh) < c.padh ||
                        ow*c.strw + kw * (1 + c.dilw) >= c.iw + c.padw ||
                        oh*c.strh + kh * (1 + c.dilh)>= c.ih + c.padh)
                        continue;

                    int ih = oh * c.strh - c.padh + kh
                            * (1 + c.dilh);
                    int iw = ow * c.strw - c.padw + kw
                            * (1 + c.dilw);
                    size_t sidx = mb * padded_ic * c.ih * c.iw
                        + g * padded_ic / c.ng * c.ih * c.iw
                        + ic * c.ih * c.iw + ih * c.iw + iw;
                    size_t didx = mb * padded_oc * c.oh * c.ow
                        + g * padded_oc / c.ng * c.oh * c.ow
                        + oc * c.oh * c.ow + oh * c.ow + ow;

                    diff_weights_data[map_index(weights_d, widx)]
                        += src_data[map_index(src_d, sidx)]
                        * diff_dst_data[map_index(dst_d, didx)];
                }
            }
        }
    });
}

void simple_net(int sparsity, int n, int mb, int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw, bool verify)
{

    int padh = 1, padw = 1;
    int strh = 1, strw = 1;
    int dilh = 0, dilw = 0;


    float *src_data = (float *) aligned_alloc(64, mb * ic * ih * iw * sizeof(float));
    float *dst_data = (float *) aligned_alloc(64, mb * oc * oh * ow * sizeof(float));
    float *wei_data = (float *) aligned_alloc(64, oc * ic * kh * kw * sizeof(float));
    float *bias_data = (float *) aligned_alloc(64, oc * sizeof(float));

    float *wei_ref_data = (float *) aligned_alloc(64, oc * ic * kh * kw * sizeof(float));

    for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        if (rand() % 100 >= sparsity) {
            dst_data[i] = i;
        } else {
            dst_data[i] = 0.0;
        }
    }

    for (size_t i = 0; i < mb * ic * ih * iw; ++i) {
        if (rand() % 100 >= sparsity) {
            src_data[i] = i;
        } else {
            src_data[i] = 0.0;
        }
    }

    auto cpu_engine = engine(engine::cpu, 0);

    auto c_src_desc = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::NhC16cw16n);
    auto c_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format::hIOw16i16o);
    auto c_dst_desc = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::NhC16nw16c);
    auto c_bias_desc = memory::desc({ oc }, memory::data_type::f32,
                                        memory::format::x);

    auto c_src_desc_f = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::NhC16cw16n);
    auto c_dst_desc_f = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::NhC16nw16c);

    auto c_src = memory({c_src_desc, cpu_engine}, src_data);
    auto c_diff_weights = memory({c_weights_desc, cpu_engine}, wei_data);
    auto c_diff_dst = memory({c_dst_desc, cpu_engine}, dst_data);
    auto c_diff_bias = memory({c_bias_desc, cpu_engine }, bias_data);

    auto c_ref_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format::hIOw16i16o);
    auto c_diff_ref_weights = memory({c_ref_weights_desc, cpu_engine}, wei_ref_data);

    std::vector<int> padR = {
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
    auto conv_bwd_weights = convolution_backward_weights(
            conv_bwd_weights_primitive_desc,
            c_src, c_diff_dst, c_diff_weights);

    std::vector<primitive> pipeline;
    pipeline.push_back(conv_bwd_weights);
    

    uint64_t min_time = UINT64_MAX;
    
    __itt_resume();
    for (int i = 0; i < n; ++i) {
        std::cout << "iter " << i << std::endl;

        uint64_t start = rdtscp();
        stream(stream::kind::eager).submit(pipeline).wait();
        uint64_t end = rdtscp();

        uint64_t time = end - start;
        if (time < min_time) {
            min_time = time;
        }
        std::cout << "min time: " << min_time << std::endl;
    }
    __itt_pause();

    if (verify) {
    
        test_convolution_sizes_t cd(mb, 1, ic, ih, iw, oc, oh, ow, kh, kw, padh, padw, strh, strw);

        compute_ref_conv_bwd_weights(cd, c_src, c_diff_dst, c_diff_ref_weights);


        /*for (size_t i = 0; i < ic * oc * kh * kw; ++i) {
            std::cout << wei_data[i] << " ";
        }
        std::cout << std::endl << std::endl;

        for (size_t i = 0; i < ic * oc * kh * kw; ++i) {
            std::cout << wei_ref_data[i] << " ";
        }
        std::cout << std::endl << std::endl;

        for (size_t i = 0; i < ic * oc * kh * kw; ++i) {
            std::cout << wei_data[i] / wei_ref_data[i] << " ";
        }
        std::cout << std::endl;*/

        check_zero_tail<float>(1, c_diff_ref_weights);

        compare_data<float>(c_diff_ref_weights, c_diff_weights);
        check_zero_tail<float>(1, c_diff_weights);

    }

}

int main(int argc, char **argv)
{
    int sparsity = 50;
    int n = 1;

    int mb = 32;
    int ic = 512, ih = 28, iw = 28;
    int oc = 512, oh = 28, ow = 28;
    int kh = 3, kw = 3;

    bool verify = VERIFY;

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
        oh = atoi(argv[8]);
    }

    if (argc > 9) {
        ow = atoi(argv[9]);
    }

    if (argc > 10) {
        kh = atoi(argv[10]);
    }

    if (argc > 11) {
        kw = atoi(argv[11]);
    }

    std::cout << "mb=" << mb << " ic=" << ic << " ih=" << ih << " iw=" << iw
        << " oc=" << oc << " oh=" << oh << " ow=" << ow << " kh=" << kh << " kw=" << kw << std::endl;

    try
    {
        simple_net(sparsity, n, mb, ic, ih, iw, oc, oh, ow, kh, kw, verify);
        std::cout << "passed" << std::endl;
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}
