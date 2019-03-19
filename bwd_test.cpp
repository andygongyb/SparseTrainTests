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

template <typename data_t_diff_dst, typename data_t_wei,
          typename data_t_acc, typename data_t_diff_src>
void compute_ref_conv_bwd_data_(const test_convolution_sizes_t &c,
        const memory &diff_src, const memory &weights, const memory &diff_dst)
{
    data_t_diff_dst *diff_dst_data = (data_t_diff_dst *)diff_dst.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();
    data_t_diff_src *diff_src_data = (data_t_diff_src *)diff_src.get_data_handle();

    const memory::desc diff_src_d = diff_src.get_primitive_desc().desc();
    const memory::desc weights_d = weights.get_primitive_desc().desc();
    const memory::desc diff_dst_d = diff_dst.get_primitive_desc().desc();

    size_t padded_ic = diff_src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = diff_dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.ic / c.ng, c.ih, c.iw,
        [&](int mb, int g, int ic, int ih, int iw) {
            size_t sidx = mb * padded_ic * c.ih * c.iw
                    + g * padded_ic / c.ng * c.ih * c.iw
                    + ic * c.ih * c.iw + ih * c.iw + iw;
            data_t_acc a = data_t_acc(0);
            for (int oc = 0; oc < c.oc / c.ng; oc++) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        if (iw + c.padw < kw * (1 + c.dilw)
                           || ih + c.padh < kh * (1 + c.dilh))
                            continue;
                        int ow = iw - kw * (1 + c.dilw) + c.padw;
                        int oh = ih - kh * (1 + c.dilh) + c.padh;
                        if (ow % c.strw != 0 || oh % c.strh != 0)
                            continue;
                        ow /= c.strw;
                        oh /= c.strh;
                        if (oh < c.oh && ow < c.ow) {
                            size_t didx = mb * padded_oc * c.oh * c.ow
                                + g * padded_oc / c.ng * c.oh * c.ow
                                + oc * c.oh * c.ow + oh * c.ow + ow;
                            size_t widx =
                                g * padded_oc / c.ng * padded_ic
                                / c.ng * c.kh * c.kw
                                + oc * padded_ic / c.ng * c.kh * c.kw
                                + ic * c.kh * c.kw + kh * c.kw + kw;

                            a += (data_t_acc)(
                                diff_dst_data[map_index(diff_dst_d, didx)]
                                * weights_data[map_index(weights_d, widx)]);
                        }
                    }
                }
            }
            diff_src_data[map_index(diff_src_d, sidx)] = (data_t_diff_src)a;
    });
}

void simple_net(int sparsity, int n, int mb, int ic, int ih, int iw, int oc, int strh,
        int strw, int kh, int kw, bool verify)
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

    float *src_ref_data = (float *) aligned_alloc(64, mb * ic * ih * iw * sizeof(float));

    size_t s = 0;

    /* initializing non-zero values for src */
    /*for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        if (rand() % 100 >= sparsity) {
            dst_data[i] = sinf((float)i) + 2.0;
        } else {
            dst_data[i] = 0.0;
            s++;
        }
    }*/

    for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        if (rand() % 100 >= sparsity) {
            dst_data[i] = i;
        } else {
            dst_data[i] = 0.0;
            //s++;
        }
    }

    //std::cout << "actuall sparsity:" << (double) s / (double) (mb * oc * oh * ow) << std::endl;

    for (size_t i = 0; i < oc * ic * kh * kw; ++i) {
        //wei_data[i] = sinf((float)i);
        wei_data[i] = i;
    }

    auto cpu_engine = engine(engine::cpu, 0);

    auto c_src_desc = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::nChw16c);
    auto c_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format::OIhw16o16i);
    auto c_dst_desc = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::nChw16c);

    auto c_src_desc_f = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::nChw16c);
    auto c_dst_desc_f = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::nChw16c);

    auto c_diff_src = memory({c_src_desc, cpu_engine}, src_data);
    auto c_weights = memory({c_weights_desc, cpu_engine}, wei_data);
    auto c_diff_dst = memory({c_dst_desc, cpu_engine}, dst_data);

    auto c_src_ref_desc = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::nChw16c);
    auto c_diff_ref_src = memory({c_src_ref_desc, cpu_engine}, src_ref_data);

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

    auto conv_bwd_data_desc = convolution_backward_data::desc(
            convolution_direct, c_src_desc, c_weights_desc, c_dst_desc,
            { strh, strw }, { dilh, dilw },
            { padh, padw }, padR, padding_kind::zero);
    auto conv_bwd_data_primitive_desc
        = convolution_backward_data::primitive_desc(
                conv_bwd_data_desc, cpu_engine, conv_primitive_desc);
    auto conv_bwd_data = convolution_backward_data(
            conv_bwd_data_primitive_desc,
            c_diff_dst, c_weights, c_diff_src);

    std::vector<primitive> pipeline;
    pipeline.push_back(conv_bwd_data);
    

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

        compute_ref_conv_bwd_data_<float, float, float, float>(
                cd, c_diff_ref_src, c_weights, c_diff_dst);


        /*for (size_t i = 0; i < mb * ic * ih * iw; ++i) {
            std::cout << src_data[i] << " ";
        }
        std::cout << std::endl << std::endl;

        for (size_t i = 0; i < mb * ic * ih * iw; ++i) {
            std::cout << src_ref_data[i] << " ";
        }
        std::cout << std::endl;*/

        check_zero_tail<float>(1, c_diff_ref_src);

        compare_data<float>(c_diff_ref_src, c_diff_src);
        check_zero_tail<float>(0, c_diff_src);

    }

}


int main(int argc, char **argv)
{
    int sparsity = 50;
    int n = 1;

    int mb = 32;
    int ic = 512, ih = 28, iw = 28;
    int oc = 512;
    int kh = 3, kw = 3;

    bool verify = VERIFY;

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
        simple_net(sparsity, n, mb, ic, ih, iw, oc, strh, strw, kh, kw, verify);
        std::cout << "passed" << std::endl;
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    return 0;
}

