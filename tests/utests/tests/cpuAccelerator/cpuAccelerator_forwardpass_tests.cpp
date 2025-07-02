#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "denseLayer.h"
#include "convLayer.h"
#include <cmath>
#include <cstdio>


TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_relu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::RELU, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, 0.0f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_forwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_sigmoid_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::SIGMOID, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::SIGMOID, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::SIGMOID, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.526226f, 0.564882f, 0.47764f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_forwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_leakyRelu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::LEAKY_RELU, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::LEAKY_RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::LEAKY_RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, -0.089411f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_forwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_softmax_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::SOFTMAX, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::SOFTMAX, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::SOFTMAX, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.334217f, 0.390641f, 0.275142f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_forwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_tanh_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::TANH, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.104616f, 0.255231f, -0.089262f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_forwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_conv_forwardpass_leakyrelu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(36, eAct_func::LEAKY_RELU, 0.01f);
    ConvLayer* curr_lyr = new ConvLayer(6, 6, 3, 3, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::LEAKY_RELU, 0.01f);

    float prev_in[36] = {0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f};

    float curr_bias[16] = {0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f};

    float fwdPassKernelMtx[9] = {0.1f, 0.6f, -0.2f, 
                                    0.3f, 0.01f, -0.4f,
                                    0.12f, 0.4f, -0.1f};

    float out[16] = {0.029889f, 0.018133f, 0.001356f, 0.046356f,
                        0.045111f, 0.009167f, -0.000023f, 0.050689f,
                        0.038956f, 0.002801f, 0.003844f, 0.051133f,
                        0.066656f, 0.005633f, 0.0008f, 0.049467f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassKernelMtx);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 16; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}
