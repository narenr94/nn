#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "baseLayer.h"
#include "denseLayer.h"
#include "convLayer.h"
#include <cmath>


TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_relu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.34f, -0.22f, 0.0f};

    curr_lyr->set_all_node_values(curr_vals);

    nxt_lyr->set_all_node_values(nxt_val);

    nxt_lyr->set_node_delta(nxt_delta[0], 0);

    nxt_lyr->set_node_delta(nxt_delta[1], 1);

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    nxt_lyr->set_all_transform_matrix_parameter(bkwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_leakyRelu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.34f, -0.22f, 0.00242f};

    curr_lyr->set_all_node_values(curr_vals);

    nxt_lyr->set_all_node_values(nxt_val);

    nxt_lyr->set_node_delta(nxt_delta[0], 0);

    nxt_lyr->set_node_delta(nxt_delta[1], 1);

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    nxt_lyr->set_all_transform_matrix_parameter(bkwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_tanh_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::TANH, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.3366f, -0.209352f, -0.37752f};

    curr_lyr->set_all_node_values(curr_vals);

    nxt_lyr->set_all_node_values(nxt_val);

    nxt_lyr->set_node_delta(nxt_delta[0], 0);

    nxt_lyr->set_node_delta(nxt_delta[1], 1);

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    nxt_lyr->set_all_transform_matrix_parameter(bkwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_sigmoid_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::SIGMOID, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.0306f, -0.037752f, -1.00672f};

    curr_lyr->set_all_node_values(curr_vals);

    nxt_lyr->set_all_node_values(nxt_val);

    nxt_lyr->set_node_delta(nxt_delta[0], 0);

    nxt_lyr->set_node_delta(nxt_delta[1], 1);

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    nxt_lyr->set_all_transform_matrix_parameter(bkwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_softmax_test)
{
    DenseLayer* prev_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(3, eAct_func::SOFTMAX, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.01296f, 0.054912f, -1.13856f};

    curr_lyr->set_all_node_values(curr_vals);

    nxt_lyr->set_all_node_values(nxt_val);

    nxt_lyr->set_node_delta(nxt_delta[0], 0);

    nxt_lyr->set_node_delta(nxt_delta[1], 1);

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    nxt_lyr->set_all_transform_matrix_parameter(bkwdPassWtMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_dense_layer();

    for(int i = 0; i < 3; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_conv_backwardpass_leakyRelu_test)
{
    DenseLayer* prev_lyr = new DenseLayer(36, eAct_func::TANH, 0.999f);
    ConvLayer* curr_lyr = new ConvLayer(6, 6, 3, 3, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(2, eAct_func::TANH, 0.999f);

    float curr_delta[16] = {0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f};

    float curr_vals[16] = {0.1f, 0.22f, -1.6f, 0.8f,
                            -0.2f, 0.4f, 0.6f, 0.3f,
                            0.1f, 0.4f, 0.2f, -0.1f,
                            0.001f, 0.9f, 0.16f, 0.4f};

    float bkwdPassKernelMtx[9] = {0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f};

    float out[36] = {-0.020000f, 0.040000f, 0.050000f, 0.050000f, 0.070000f, 0.010000f, 
-0.040000f, 0.080000f, 0.100000f, 0.100000f, 0.140000f, 0.020000f, 
-0.060000f, 0.120000f, 0.150000f, 0.150000f, 0.210000f, 0.030000f, 
-0.060000f, 0.120000f, 0.150000f, 0.150000f, 0.210000f, 0.030000f, 
-0.040000f, 0.080000f, 0.100000f, 0.100000f, 0.140000f, 0.020000f,
-0.020000f, 0.040000f, 0.050000f, 0.050000f, 0.070000f, 0.010000f};

    curr_lyr->set_all_node_values(curr_vals);

    for(int i = 0; i < 36; i++)
    {
        curr_lyr->set_node_delta(curr_delta[i], i);
    }

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    curr_lyr->set_all_transform_matrix_parameter(bkwdPassKernelMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(prev_lyr);

    cpuAcc->do_backwardpass_conv_layer(6, 6, 3, 3);

    for(int i = 0; i < 36; i++)
    {
        float roundedValue = std::round(prev_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

