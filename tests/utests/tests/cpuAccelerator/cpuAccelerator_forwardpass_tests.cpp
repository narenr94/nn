#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "inputLayer.h"
#include "denseLayer.h"
#include "convLayer.h"
#include "poolingLayer.h"
#include <cmath>
#include <cstdio>


TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_relu_test)
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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::RELU, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};
    std::vector<float> prev_in_vec(prev_in, prev_in + 2);

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, 0.0f};

    prev_lyr->set_all_node_values(prev_in_vec);

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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::SIGMOID, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::SIGMOID, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::SIGMOID, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};
    std::vector<float> prev_in_vec(prev_in, prev_in + 2);

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.526226f, 0.564882f, 0.47764f};

    prev_lyr->set_all_node_values(prev_in_vec);

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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::LEAKY_RELU, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::LEAKY_RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};
    std::vector<float> prev_in_vec(prev_in, prev_in + 2);

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, -0.089411f};

    prev_lyr->set_all_node_values(prev_in_vec);

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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::SOFTMAX, 0.999f);
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::SOFTMAX, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::SOFTMAX, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};
    std::vector<float> prev_in_vec(prev_in, prev_in + 2);

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.334217f, 0.390641f, 0.275142f};

    prev_lyr->set_all_node_values(prev_in_vec);

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
    DenseLayer* curr_lyr = new DenseLayer(dims2, eAct_func::TANH, 0.999f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::TANH, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};
    std::vector<float> prev_in_vec(prev_in, prev_in + 2);

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.104616f, 0.255231f, -0.089262f};

    prev_lyr->set_all_node_values(prev_in_vec);

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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::LEAKY_RELU, 0.01f);

    float prev_in[36] = {0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f};

    std::vector<float> prev_in_vec(prev_in, prev_in + 36);

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

    prev_lyr->set_all_node_values(prev_in_vec);

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

TEST(CPU_ACC_TESTS, cpuAccelerator_conv_forwardpass_multiKernel_leakyrelu_test)
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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::LEAKY_RELU, 0.01f);

    float prev_in[36] = {0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f};
    
    std::vector<float> prev_in_vec(prev_in, prev_in + 36);

    float curr_bias[48] = {0.1f, 0.1f, 0.1f, 0.1f,
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

    float fwdPassKernelMtx[27] = {0.1f, 0.6f, -0.2f, 
                                    0.3f, 0.01f, -0.4f,
                                    0.12f, 0.4f, -0.1f,
                                    0.2f, 0.3f, 0.4f, 
                                    0.5f, 0.01f, -0.2f,
                                    -0.3f, 40.4f, -0.5f,
                                    -0.1f, -0.6f, 0.2f, 
                                    -0.3f, -0.01f, 0.4f,
                                    -0.12f, -0.4f, 0.1f};

    // float out[16] = {0.029889f, 0.018133f, 0.001356f, 0.046356f,
    //                     0.045111f, 0.009167f, -0.000023f, 0.050689f,
    //                     0.038956f, 0.002801f, 0.003844f, 0.051133f,
    //                     0.066656f, 0.005633f, 0.0008f, 0.049467f};

    float out[48] = {0.029889f, 0.018133f, 0.001356f, 0.046356f,
                        0.045111f, 0.009167f, -2.3e-05f, 0.050689f,
                        0.038956f, 0.002801f, 0.003844f, 0.051133f,
                        0.066656f, 0.005633f, 0.0008f, 0.049467f,
                        0.074556f, 0.065667f, 0.951667f, 1.81688905f,
                        0.9164f, 0.005389f, 0.946756f, 1.81933403f,
                        0.068644f, 0.060334f, 0.947622f, 1.82144499f,
                        0.930945f, 0.456933f, 0.496033f, 1.82666695f,
                        -7.7e-05f, 0.004089f, 0.020867f, -0.000241f,
                        -0.000229f, 0.013056f, 0.024531f, -0.000285f,
                        -0.000167f, 0.019421f, 0.018378f, -0.000289f,
                        -0.000444f, 0.016589f, 0.021422f, -0.000272f};

    prev_lyr->set_all_node_values(prev_in_vec);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassKernelMtx);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 48; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}


TEST(CPU_ACC_TESTS, cpuAccelerator_pooling_forwardpass_max)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 2;
    dims1.unOutputColumns = 10;
    dims1.unOutputRows = 10;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 10;
    dims2.unInputRows = 10;
    dims2.unNoTransformParameterMtx = 2;
    dims2.unOutputColumns = 4;
    dims2.unOutputRows = 4;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    PoolingLayer* curr_lyr = new PoolingLayer(dims2, ePooling_type::MAX);

    float in[100] = {   
                        0,1,2,      3,4,5,      6,7,8,      9,
                        10,11,12,   13,14,15,   16,17,18,   19,
                        20,21,22,   23,24,25,   26,27,28,   29,

                        30,31,32,   33,34,35,   36,37,38,   39,
                        40,41,42,   43,44,45,   46,47,48,   49,

                        50,51,52,   53,54,55,   56,57,58,   59,
                        60,61,62,   63,64,65,   66,67,68,   69,
                        70,71,72,   73,74,75,   76,77,78,   79,
                        
                        80,81,82,   83,84,85,   86,87,88,   89,
                        90,91,92,   93,94,95,   96,97,98,   99
                    };

    float out[16] = {
                        22, 25, 28, 29,
                        42, 45, 48, 49,
                        72, 75, 78, 79,
                        92, 95, 98, 99

                    };

    std::vector<float> prev_in_vec(in, in + 100);

    prev_lyr->set_all_node_values(prev_in_vec);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 16; i++)
    {
        EXPECT_EQ(curr_lyr->get_node_value_idx(i), out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_pooling_forwardpass_avg)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 2;
    dims1.unOutputColumns = 10;
    dims1.unOutputRows = 10;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 10;
    dims2.unInputRows = 10;
    dims2.unNoTransformParameterMtx = 2;
    dims2.unOutputColumns = 4;
    dims2.unOutputRows = 4;
    dims2.unTransformParametersColumns = 3;
    dims2.unTransformParametersRows = 3;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    PoolingLayer* curr_lyr = new PoolingLayer(dims2, ePooling_type::AVERAGE);

    float in[100] = {   
                        0,1,2,      3,4,5,      6,7,8,      9,
                        10,11,12,   13,14,15,   16,17,18,   19,
                        20,21,22,   23,24,25,   26,27,28,   29,

                        30,31,32,   33,34,35,   36,37,38,   39,
                        40,41,42,   43,44,45,   46,47,48,   49,

                        50,51,52,   53,54,55,   56,57,58,   59,
                        60,61,62,   63,64,65,   66,67,68,   69,
                        70,71,72,   73,74,75,   76,77,78,   79,
                        
                        80,81,82,   83,84,85,   86,87,88,   89,
                        90,91,92,   93,94,95,   96,97,98,   99
                    };

    float out[16] = {
                        11, 14, 17, 19,
                        36, 39, 42, 44,
                        61, 64, 67, 69,
                        86, 89, 92, 94

                    };

    std::vector<float> prev_in_vec(in, in + 100);

    prev_lyr->set_all_node_values(prev_in_vec);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 16; i++)
    {
        EXPECT_EQ(curr_lyr->get_node_value_idx(i), out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_pooling_forwardpass_gae)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 2;
    dims1.unOutputColumns = 10;
    dims1.unOutputRows = 10;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 10;
    dims2.unInputRows = 10;
    dims2.unNoTransformParameterMtx = 2;
    dims2.unOutputColumns = 2;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 10;
    dims2.unTransformParametersRows = 5;

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    PoolingLayer* curr_lyr = new PoolingLayer(dims2, ePooling_type::AVERAGE);

    float in[100] = {   
                        0,1,2,      3,4,5,      6,7,8,      9,
                        10,11,12,   13,14,15,   16,17,18,   19,
                        20,21,22,   23,24,25,   26,27,28,   29,

                        30,31,32,   33,34,35,   36,37,38,   39,
                        40,41,42,   43,44,45,   46,47,48,   49,

                        50,51,52,   53,54,55,   56,57,58,   59,
                        60,61,62,   63,64,65,   66,67,68,   69,
                        70,71,72,   73,74,75,   76,77,78,   79,
                        
                        80,81,82,   83,84,85,   86,87,88,   89,
                        90,91,92,   93,94,95,   96,97,98,   99
                    };

    float out[2] = {
                        24.5, 74.5
                    };

    std::vector<float> prev_in_vec(in, in + 100);

    prev_lyr->set_all_node_values(prev_in_vec);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 2; i++)
    {
        EXPECT_EQ(curr_lyr->get_node_value_idx(i), out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
}


TEST(CPU_ACC_TESTS, cpuAccelerator_conv_forwardpass_multiKernel_multiInput_leakyrelu_test)
{
    sLayer_Dimensions dims1, dims2, dims3;
    
    dims1.unInputColumns = 0;
    dims1.unInputRows = 0;
    dims1.unNoTransformParameterMtx = 3;
    dims1.unOutputColumns = 6;
    dims1.unOutputRows = 18;
    dims1.unTransformParametersColumns = 0;
    dims1.unTransformParametersRows = 0;

    dims2.unInputColumns = 6;
    dims2.unInputRows = 18;
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

    InputLayer* prev_lyr = new InputLayer(dims1, eAct_func::LEAKY_RELU, 0.01f);
    ConvLayer* curr_lyr = new ConvLayer(dims2, eAct_func::LEAKY_RELU, 0.01f);
    DenseLayer* nxt_lyr = new DenseLayer(dims3, eAct_func::LEAKY_RELU, 0.01f);

    float prev_in[108] = {0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f,

                            0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f,

                            0.5f, 0.2f, 0.1f, 0.11f, 0.4f, 0.26f,
                            0.1f, 0.2f, 0.1f, 0.1f, 0.4f, 0.2f,
                            0.2f, 0.01f, 0.01f, 0.21f, 0.4f, 0.21f,
                            0.5f, 0.2f, 0.001f, 0.21f, 0.4f, 0.22f,
                            0.6f, 0.01f, 0.01f, 0.21f, 0.4f, 0.23f,
                            0.7f, 0.2f, 0.1f, 0.11f, 0.4f, 0.2f};

    float curr_bias[48] = {0.1f, 0.1f, 0.1f, 0.1f,
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

    float fwdPassKernelMtx[27] = {0.1f, 0.6f, -0.2f, 
                                    0.3f, 0.01f, -0.4f,
                                    0.12f, 0.4f, -0.1f,

                                    0.2f, 0.3f, 0.4f, 
                                    0.5f, 0.01f, -0.2f,
                                    -0.3f, 40.4f, -0.5f,

                                    -0.1f, -0.6f, 0.2f, 
                                    -0.3f, -0.01f, 0.4f,
                                    -0.12f, -0.4f, 0.1f};

    // float out[16] = {0.029889f, 0.018133f, 0.001356f, 0.046356f,
    //                     0.045111f, 0.009167f, -0.000023f, 0.050689f,
    //                     0.038956f, 0.002801f, 0.003844f, 0.051133f,
    //                     0.066656f, 0.005633f, 0.0008f, 0.049467f};

    float out[48] = {0.029889f, 0.018133f, 0.001356f, 0.046356f,
                        0.045111f, 0.009167f, -2.3e-05f, 0.050689f,
                        0.038956f, 0.002801f, 0.003844f, 0.051133f,
                        0.066656f, 0.005633f, 0.0008f, 0.049467f,
                        0.074556f, 0.065667f, 0.951667f, 1.81688905f,
                        0.9164f, 0.005389f, 0.946756f, 1.81933403f,
                        0.068644f, 0.060334f, 0.947622f, 1.82144499f,
                        0.930945f, 0.456933f, 0.496033f, 1.82666695f,
                        -7.7e-05f, 0.004089f, 0.020867f, -0.000241f,
                        -0.000229f, 0.013056f, 0.024531f, -0.000285f,
                        -0.000167f, 0.019421f, 0.018378f, -0.000289f,
                        -0.000444f, 0.016589f, 0.021422f, -0.000272f};

    std::vector<float> prev_in_vec(prev_in, prev_in + 108);

    prev_lyr->set_all_node_values(prev_in_vec);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->set_all_transform_matrix_parameter(fwdPassKernelMtx);

    curr_lyr->do_forwardpass_to_current_layer();

    for(int i = 0; i < 48; i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_value_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete prev_lyr;
    delete curr_lyr;
    delete nxt_lyr;
}