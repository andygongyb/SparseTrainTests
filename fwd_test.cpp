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

/*struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};*/

template <typename data_t_src, typename data_t_wei,
          typename data_t_acc, typename data_t_dst>
void compute_ref_conv_fwd_(const test_convolution_sizes_t &c,
        const memory::desc &src_d,
        const memory::desc &weights_d,
        const memory::desc &bias_d,
        const memory::desc &dst_d,
        const memory &src,
        const memory &weights,
        const memory &bias,
        const memory &dst)
{
    const bool w_bias = bias_d.data.format != memory::format::format_undef;
    data_t_src *src_data = (data_t_src *)src.get_data_handle();
    data_t_wei *weights_data = (data_t_wei *)weights.get_data_handle();

    data_t_dst *bias_data = w_bias ? (data_t_dst *)bias.get_data_handle() : nullptr;
    data_t_dst *dst_data = (data_t_dst *)dst.get_data_handle();

    size_t padded_ic = src_d.data.layout_desc.blocking.padding_dims[1];
    size_t padded_oc = dst_d.data.layout_desc.blocking.padding_dims[1];

    mkldnn::impl::parallel_nd(c.mb, c.ng, c.oc / c.ng, c.oh, c.ow,
        [&](int n, int g, int oc, int oh, int ow) {
            data_t_acc a = 0;
            for (int ic = 0; ic < c.ic / c.ng; ic++) {
                for (int kh = 0; kh < c.kh; kh++) {
                    for (int kw = 0; kw < c.kw; kw++) {
                        int iw = ow * c.strw
                              - c.padw + kw * (1 + c.dilw);
                        int ih = oh * c.strh
                              - c.padh + kh * (1 + c.dilh);
                        if (iw < 0 || iw >= c.iw) continue;
                        if (ih < 0 || ih >= c.ih) continue;
                        size_t iidx = n * padded_ic * c.ih * c.iw
                            + g * padded_ic / c.ng * c.ih * c.iw
                            + ic * c.ih * c.iw + ih * c.iw + iw;
                        size_t widx = g * padded_oc / c.ng * padded_ic
                            / c.ng * c.kh * c.kw
                            + oc * padded_ic / c.ng * c.kh * c.kw
                            + ic * c.kh * c.kw + kh * c.kw + kw;
                        a += ((data_t_acc)
                            src_data[map_index(src_d, iidx)])
                            *  weights_data[map_index(
                            weights_d, widx)];
                    }
                }
            }

            float a_fp = (float)a;

            a_fp += (float)(bias_data
                ?  bias_data[map_index(bias_d, g * c.oc / c.ng + oc)] : 0);

            size_t oidx = n * padded_oc * c.oh * c.ow
                     + g * padded_oc / c.ng * c.oh * c.ow
                     + oc * c.oh * c.ow + oh * c.ow + ow;
            dst_data[map_index(dst_d, oidx)] = (data_t_dst)a_fp;
        }
    );
}

void simple_net(int sparsity, int n, int mb, int ic, int ih, int iw, int oc, int strh, int strw, int kh, int kw)
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

    float *dst_ref_data = (float *) aligned_alloc(64, mb * oc * oh * ow * sizeof(float));

    size_t s = 0;

    /* initializing non-zero values for src */
    for (size_t i = 0; i < mb * ic * ih * iw; ++i) {
        if (rand() % 100 >= sparsity) {
            src_data[i] = i;
        } else {
            src_data[i] = 0.0;
            s++;
        }
    }

    for (size_t i = 0; i < oc * ic * kh * kw; ++i) {
        wei_data[i] = i;
    }

    for (size_t i = 0; i < oc; ++i) {
        bias_data[i] = i;
    }

    std::cout << "actuall sparsity:" << (double) s / (double) (mb * ic * ih * iw) << std::endl;

    auto cpu_engine = engine(engine::cpu, 0);

    auto c_src_desc = memory::desc({ mb, ic, ih, iw }, memory::data_type::f32,
                                        memory::format::nChw16c);
    auto c_weights_desc = memory::desc({ oc, ic, kh, kw }, memory::data_type::f32,
                                        memory::format::OIhw16i16o);
    auto c_dst_desc = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::nChw16c);

    auto c_bias_desc = memory::desc({ oc }, memory::data_type::f32,
                                        memory::format::x);

    auto c_src = memory({c_src_desc, cpu_engine}, src_data);
    auto c_weights = memory({c_weights_desc, cpu_engine}, wei_data);
    auto c_dst = memory({c_dst_desc, cpu_engine}, dst_data);
    auto c_bias = memory({c_bias_desc, cpu_engine }, bias_data);


    auto c_dst_ref_desc = memory::desc({ mb, oc, oh, ow }, memory::data_type::f32,
                                        memory::format::nChw16c);

    auto c_dst_ref = memory({c_dst_ref_desc, cpu_engine}, dst_ref_data);

    std::vector<int> padR = {
        right_padding_(ih, oh, kh, padh, strh, dilh),
        right_padding_(iw, ow, kw, padw, strw, dilw)
    };

    auto conv_desc = convolution_forward::desc(
            prop_kind::forward_training, convolution_direct, c_src_desc,
            c_weights_desc, c_bias_desc, c_dst_desc,
            { strh, strw }, { dilh, dilw },
            { padh, padw }, padR, padding_kind::zero);
    auto conv_primitive_desc = convolution_forward::primitive_desc(
            conv_desc, cpu_engine);

    auto conv_fwd = convolution_forward(conv_primitive_desc, c_src,
            c_weights, c_bias, c_dst);

    std::vector<primitive> pipeline;
    pipeline.push_back(conv_fwd);
    
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

    size_t padded_ic = c_src_desc.data.layout_desc.blocking.padding_dims[1];

    size_t iidx = 0 * padded_ic * ih * iw + 0 * ih * iw + 0 * iw + 1;
    /*size_t widx = g * padded_oc / c.ng * padded_ic
        / c.ng * c.kh * c.kw
        + oc * padded_ic / c.ng * kh * kw
        + 0 * kh * kw + 0 * kw + 0;*/

    /*std::cout << "padded ic: " << padded_ic << " idx: " << map_index(c_src_desc, iidx) << std::endl;

    test_convolution_sizes_t cd(mb, 1, ic, ih, iw, oc, oh, ow, kh, kw, padh, padw, strh, strw);

    compute_ref_conv_fwd_<float, float, float, float>(
            cd, c_src_desc, c_weights_desc, c_bias_desc, c_dst_ref_desc,
            c_src, c_weights, c_bias, c_dst_ref);

    for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        std::cout << dst_data[i] << " ";
    }
    std::cout << std::endl << std::endl;

    for (size_t i = 0; i < mb * oc * oh * ow; ++i) {
        std::cout << dst_ref_data[i] << " ";
    }
    std::cout << std::endl;

    check_zero_tail<float>(1, c_dst_ref);

    compare_data<float>(c_dst_ref, c_dst);
    check_zero_tail<float>(0, c_dst);*/
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
