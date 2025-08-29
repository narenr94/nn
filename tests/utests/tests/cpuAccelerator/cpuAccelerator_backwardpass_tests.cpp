#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "baseLayer.h"
#include "denseLayer.h"
#include "convLayer.h"
#include "inputLayer.h"
#include "poolingLayer.h"
#include <cmath>


TEST(CPU_ACC_TESTS, cpuAccelerator_dense_backwardpass_relu_test)
{
    sLayer_Dimensions dims1, dims2, dims3;

    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 2;
    dims1.unOutputRows = 1;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 2;
    dims2.unInputRows = 1;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 2;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 1;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.34f, -0.22f, 0.0f};

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 3);
    std::vector<float> nxt_val_vec(nxt_val, nxt_val + 2);

    curr_lyr->set_all_node_values(curr_vals_vec);

    nxt_lyr->set_all_node_values(nxt_val_vec);

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
    sLayer_Dimensions dims1, dims2, dims3;

    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 2;
    dims1.unOutputRows = 1;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 2;
    dims2.unInputRows = 1;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 2;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 1;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.34f, -0.22f, 0.00242f};

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 3);
    std::vector<float> nxt_val_vec(nxt_val, nxt_val + 2);

    curr_lyr->set_all_node_values(curr_vals_vec);

    nxt_lyr->set_all_node_values(nxt_val_vec);

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
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 2;
    dims1.unOutputRows = 1;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 2;
    dims2.unInputRows = 1;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 2;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 1;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::TANH, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.3366f, -0.209352f, -0.37752f};

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 3);
    std::vector<float> nxt_val_vec(nxt_val, nxt_val + 2);

    curr_lyr->set_all_node_values(curr_vals_vec);

    nxt_lyr->set_all_node_values(nxt_val_vec);

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
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 2;
    dims1.unOutputRows = 1;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 2;
    dims2.unInputRows = 1;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 2;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 1;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::SIGMOID, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {-0.0306f, -0.037752f, -1.00672f};

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 3);
    std::vector<float> nxt_val_vec(nxt_val, nxt_val + 2);

    curr_lyr->set_all_node_values(curr_vals_vec);

    nxt_lyr->set_all_node_values(nxt_val_vec);

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
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 2;
    dims1.unOutputRows = 1;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 2;
    dims2.unInputRows = 1;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 2;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 1;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::SOFTMAX, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float nxt_val[2] = {0.5f, 0.2f};

    float nxt_delta[2] = {0.2f, -0.6f};

    float curr_vals[3] = {0.1f, 0.22f, -1.6f};

    float bkwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.01296f, 0.054912f, -1.13856f};

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 3);
    std::vector<float> nxt_val_vec(nxt_val, nxt_val + 2);

    curr_lyr->set_all_node_values(curr_vals_vec);

    nxt_lyr->set_all_node_values(nxt_val_vec);

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
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 6;
    dims1.unOutputRows = 6;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 6;
    dims2.unInputRows = 6;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 4;
    dims2.unOutputRows = 4;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    dims3.unInputColumns = 4;
    dims3.unInputRows = 4;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 16;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

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

    std::vector<float> curr_vals_vec(curr_vals, curr_vals + 16);
    curr_lyr->set_all_node_values(curr_vals_vec);

    for(int i = 0; i < 36; i++)
    {
        curr_lyr->set_node_delta(curr_delta[i], i);
    }

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    curr_lyr->set_all_transform_matrix_parameter(bkwdPassKernelMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(prev_lyr);

    cpuAcc->do_backwardpass_conv_layer();

    for(int i = 0; i < 36; i++)
    {
        float roundedValue = std::round(prev_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_conv_backwardpass_multiKernel_leakyRelu_test)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 6;
    dims1.unOutputRows = 6;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 6;
    dims2.unInputRows = 6;
    dims2.unNoTransformParameterMtx = 3;
    dims2.unOutputColumns = 4;
    dims2.unOutputRows = 12;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    dims3.unInputColumns = 4;
    dims3.unInputRows = 12;
    dims3.unNoTransformParameterMtx = 1;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 1;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 48;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float curr_delta[48] = {0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f};

    
    float bkwdPassKernelMtx[27] = {0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f,
                                    0.1f, 0.6f, -0.2f};

    float out[36] = {-0.06f, 0.12f, 0.15f, 0.15f, 0.21f, 0.03f, 
                        -0.12f, 0.24f, 0.3f, 0.3f, 0.42f, 0.06f, 
                        -0.18f, 0.36f, 0.45f, 0.45f, 0.63f, 0.09f, 
                        -0.18f, 0.36f, 0.45f, 0.45f, 0.63f, 0.09f, 
                        -0.12f, 0.24f, 0.3f, 0.3f, 0.42f, 0.06f,
                        -0.06f, 0.12f, 0.15f, 0.15f, 0.21f, 0.03f};

    for(int i = 0; i < 48; i++)
    {
        curr_lyr->set_node_delta(curr_delta[i], i);
    }

    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);

    curr_lyr->set_all_transform_matrix_parameter(bkwdPassKernelMtx);

    CpuAccelerator* cpuAcc = new CpuAccelerator(prev_lyr);

    cpuAcc->do_backwardpass_conv_layer();

    for(int i = 0; i < 36; i++)
    {
        float roundedValue = std::round(prev_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete curr_lyr;
    delete nxt_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_pooling_backwardpass_max)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 5;
    dims1.unOutputRows = 5;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 5;
    dims2.unInputRows = 5;
    dims2.unNoTransformParameterMtx = 3;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 9;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 9;
    dims3.unNoTransformParameterMtx = 3;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 6;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 2;

    ePooling_type pooling_type = ePooling_type::MAX;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    PoolingLayer* nxt_lyr = new PoolingLayer(dims3, pooling_type);


    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);


    float curr_delta[12] = {0.5f, 0.5f,
                            0.5f, 0.5f,
                            0.5f, 0.5f,
                            0.5f, 0.5f,
                            0.5f, 0.5f,
                            0.5f, 0.5f};

    float prev_value[27] = {
        0.1, 0.2,       0.3,
        0.4, 0.5,       0.6,

        0.7, 0.8,       0.9,

        0.11, 0.12,     0.13,
        0.14, 0.15,     0.16,

        0.17, 0.18,     0.19,

        0.21, 0.22,     0.23,
        0.24, 0.25,     0.26,

        0.27, 0.28,     0.29
    };

    float out[27] = {
        0.0, 0.0,       0.0,
        0.0, 0.5,       0.5,

        0.0, 0.5,       0.5,

        0.0, 0.0,       0.0,
        0.0, 0.5,       0.5,

        0.0, 0.5,       0.5,

        0.0, 0.0,       0.0,
        0.0, 0.5,       0.5,

        0.0, 0.5,       0.5
    };

    std::vector<float> prev_vals_vec(prev_value, prev_value + 27);
    curr_lyr->set_all_node_values(prev_vals_vec);

    for(uint i = 0; i < 12; i++)
    {
        nxt_lyr->set_node_delta(curr_delta[i], i);
    }
    
    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_pooling_layer(pooling_type);

    for(uint i = 0; i < 27; i++)
    {
        EXPECT_EQ(curr_lyr->get_node_delta_idx(i), out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;

}


TEST(CPU_ACC_TESTS, cpuAccelerator_pooling_backwardpass_avg)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 1;
    dims1.unOutputColumns = 5;
    dims1.unOutputRows = 5;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 5;
    dims2.unInputRows = 5;
    dims2.unNoTransformParameterMtx = 3;
    dims2.unOutputColumns = 3;
    dims2.unOutputRows = 9;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    dims3.unInputColumns = 3;
    dims3.unInputRows = 9;
    dims3.unNoTransformParameterMtx = 3;
    dims3.unOutputColumns = 2;
    dims3.unOutputRows = 6;
    dims3.unTransformParametersColumns = 2;
    dims3.unTransformParametersRows = 2;

    ePooling_type pooling_type = ePooling_type::AVERAGE;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::TANH, 0.999f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    PoolingLayer* nxt_lyr = new PoolingLayer(dims3, pooling_type);


    prev_lyr->SetPreviousNextLayers(nullptr, curr_lyr);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    nxt_lyr->SetPreviousNextLayers(curr_lyr, nullptr);


    float curr_delta[12] = {0.5f, 0.6f,
                            0.7f, 0.8f,
                            0.9f, 0.1f,
                            0.55f, 0.56f,
                            0.57f, 0.58f,
                            0.59f, 0.6f};

    float prev_value[27] = {
        0.1, 0.2,       0.3,
        0.4, 0.5,       0.6,

        0.7, 0.8,       0.9,

        0.11, 0.12,     0.13,
        0.14, 0.15,     0.16,

        0.17, 0.18,     0.19,

        0.21, 0.22,     0.23,
        0.24, 0.25,     0.26,

        0.27, 0.28,     0.29
    };

    float out[27] = {
        0.125, 0.125,       0.3,
        0.125, 0.125,       0.3,

        0.35, 0.35,         0.8,

        0.225, 0.225,       0.05,
        0.225, 0.225,       0.05,

        0.275, 0.275,       0.56,

        0.1425, 0.1425,     0.29,
        0.1425, 0.1425,     0.29,

        0.295, 0.295,       0.6
    };

    std::vector<float> prev_vals_vec(prev_value, prev_value + 27);

    curr_lyr->set_all_node_values(prev_vals_vec);
    for(uint i = 0; i < 12; i++)
    {
        nxt_lyr->set_node_delta(curr_delta[i], i);
    }
    
    CpuAccelerator* cpuAcc = new CpuAccelerator(curr_lyr);

    cpuAcc->do_backwardpass_pooling_layer(pooling_type);

    for(uint i = 0; i < 27; i++)
    {
        EXPECT_EQ(curr_lyr->get_node_delta_idx(i), out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;

}
