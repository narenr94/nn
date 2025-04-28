#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "nn_layer.h"
#include "nn_l2l_weight_matrix.h"
#include <cmath>

//Loss functions
#include "baseLossFunction.h"
#include "meanSquaredError.h"
#include "meanAbsoluteError.h"
#include "huberLoss.h"
#include "binaryCrossEntropyLoss.h"
#include "competitiveCrossEntropyLoss.h"


TEST(CPU_ACC_TESTS, cpuAccelerator_dense_forwardpass_relu_test)
{
    nn_layer* prev_lyr = new nn_layer(2, eAct_func::RELU, 0.999f);
    nn_layer* curr_lyr = new nn_layer(3, eAct_func::RELU, 0.999f);
    nn_layer* nxt_lyr = new nn_layer(2, eAct_func::RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, 0.0f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->GetWeightMatrix()->set_all_weight(fwdPassWtMtx);

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
    nn_layer* prev_lyr = new nn_layer(2, eAct_func::SIGMOID, 0.999f);
    nn_layer* curr_lyr = new nn_layer(3, eAct_func::SIGMOID, 0.999f);
    nn_layer* nxt_lyr = new nn_layer(2, eAct_func::SIGMOID, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.526226f, 0.564882f, 0.47764f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->GetWeightMatrix()->set_all_weight(fwdPassWtMtx);

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
    nn_layer* prev_lyr = new nn_layer(2, eAct_func::LEAKY_RELU, 0.999f);
    nn_layer* curr_lyr = new nn_layer(3, eAct_func::LEAKY_RELU, 0.999f);
    nn_layer* nxt_lyr = new nn_layer(2, eAct_func::LEAKY_RELU, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.105f, 0.261f, -0.089411f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->GetWeightMatrix()->set_all_weight(fwdPassWtMtx);

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
    nn_layer* prev_lyr = new nn_layer(2, eAct_func::SOFTMAX, 0.999f);
    nn_layer* curr_lyr = new nn_layer(3, eAct_func::SOFTMAX, 0.999f);
    nn_layer* nxt_lyr = new nn_layer(2, eAct_func::SOFTMAX, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.334217f, 0.390641f, 0.275142f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->GetWeightMatrix()->set_all_weight(fwdPassWtMtx);

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
    nn_layer* prev_lyr = new nn_layer(2, eAct_func::TANH, 0.999f);
    nn_layer* curr_lyr = new nn_layer(3, eAct_func::TANH, 0.999f);
    nn_layer* nxt_lyr = new nn_layer(2, eAct_func::TANH, 0.999f);

    float prev_in[2] = {0.5f, 0.2f};

    float curr_bias[3] = {0.1f, 0.22f, 0.001f};

    float fwdPassWtMtx[6] = {0.1f, 0.6f, -0.2f, 0.3f, 0.01f, -0.4f};

    float out[3] = {0.104616f, 0.255231f, -0.089262f};

    prev_lyr->set_all_node_values(prev_in);

    curr_lyr->set_all_node_biases(curr_bias);

    curr_lyr->SetPreviousNextLayers(prev_lyr, nxt_lyr);

    curr_lyr->GetWeightMatrix()->set_all_weight(fwdPassWtMtx);

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

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_outputLayer_backwardpass_MAE)
{
    nn_layer* curr_lyr = new nn_layer(5, eAct_func::TANH, 0.999f);
    curr_lyr->SetPreviousNextLayers(nullptr, nullptr);

    float fExpOut[5] = {0.1f, 0.22f, 0.001f, 0.0f, 1.0f};

    float fActOut[5] = {0.5f, 0.77f, 0.0f, 1.0f, 0.0f};

    float out[5] = {0.15f, 0.08142f, -0.2f, 0.0f, -0.2f};

    curr_lyr->set_all_node_values(fActOut);

    BaseLossFunction* mae = new MeanAbsoluteError(curr_lyr);

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    for(uint i = 0; i < curr_lyr->get_num_nodes(); i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    
    delete curr_lyr;
    delete mae;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_outputLayer_backwardpass_MSE)
{
    nn_layer* curr_lyr = new nn_layer(5, eAct_func::TANH, 0.999f);
    curr_lyr->SetPreviousNextLayers(nullptr, nullptr);

    float fExpOut[5] = {0.1f, 0.22f, 0.001f, 0.0f, 1.0f};

    float fActOut[5] = {0.5f, 0.77f, 0.0f, 1.0f, 0.0f};

    float out[5] = {0.12f, 0.089562f, -0.0004f, 0.0f, -0.4f};

    curr_lyr->set_all_node_values(fActOut);

    BaseLossFunction* mae = new MeanSquaredError(curr_lyr);

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    for(uint i = 0; i < curr_lyr->get_num_nodes(); i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    
    delete curr_lyr;
    delete mae;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_outputLayer_backwardpass_HUBER)
{
    nn_layer* curr_lyr = new nn_layer(5, eAct_func::TANH, 0.999f);
    curr_lyr->SetPreviousNextLayers(nullptr, nullptr);

    float fExpOut[5] = {0.1f, 0.22f, 0.001f, 0.0f, 1.0f};

    float fActOut[5] = {0.5f, 0.77f, 0.0f, 1.0f, 0.0f};

    float out[5] = {0.3f, 0.223905f, -0.001f, 0.0f, -1.0f};

    curr_lyr->set_all_node_values(fActOut);

    BaseLossFunction* mae = new HuberLoss(curr_lyr);

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    for(uint i = 0; i < curr_lyr->get_num_nodes(); i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    
    delete curr_lyr;
    delete mae;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_outputLayer_backwardpass_CCE)
{
    nn_layer* curr_lyr = new nn_layer(5, eAct_func::TANH, 0.999f);
    curr_lyr->SetPreviousNextLayers(nullptr, nullptr);

    float fExpOut[5] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f};

    float fActOut[5] = {0.5f, 0.77f, 0.01f, 1.0f, 0.01f};

    float out[5] = {-1.5f, 0.0f, -99.99f, 0.0f, 0.0f};

    curr_lyr->set_all_node_values(fActOut);

    BaseLossFunction* mae = new CompetitiveCrossEntropyLoss(curr_lyr);

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    for(uint i = 0; i < curr_lyr->get_num_nodes(); i++)
    {
        float roundedValue = std::round(curr_lyr->get_node_delta_idx(i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    
    delete curr_lyr;
    delete mae;
}

TEST(CPU_ACC_TESTS, cpuAccelerator_dense_outputLayer_backwardpass_BCE)
{
    nn_layer* curr_lyr = new nn_layer(1, eAct_func::SIGMOID, 0.999f);
    curr_lyr->SetPreviousNextLayers(nullptr, nullptr);
    BaseLossFunction* mae = new CompetitiveCrossEntropyLoss(curr_lyr);

    float fExpOut[1] = {1.0f};

    float fActOut[1] = {0.5f};

    float out[1] = {-0.5f};

    curr_lyr->set_all_node_values(fActOut);    

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    float roundedValue = std::round(curr_lyr->get_node_delta_idx(0) * 1000000.0f) / 1000000.0f;
    EXPECT_EQ(roundedValue, out[0]);

    fExpOut[1] = {0.0f};

    fActOut[1] = {0.5f};

    out[0] = {-0.999999f};

    curr_lyr->set_all_node_values(fActOut);    

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    roundedValue = std::round(curr_lyr->get_node_delta_idx(0) * 1000000.0f) / 1000000.0f;
    EXPECT_EQ(roundedValue, out[0]);

    fExpOut[1] = {0.0f};

    fActOut[1] = {-0.5f};

    out[0] = {-0.999999f};

    curr_lyr->set_all_node_values(fActOut);    

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    roundedValue = std::round(curr_lyr->get_node_delta_idx(0) * 1000000.0f) / 1000000.0f;
    EXPECT_EQ(roundedValue, out[0]);

    fExpOut[1] = {1.0f};

    fActOut[1] = {1.5f};

    out[0] = {0.0f};

    curr_lyr->set_all_node_values(fActOut);    

    curr_lyr->do_backwardpass_to_previous_layer_output_layer(fExpOut, mae);

    roundedValue = std::round(curr_lyr->get_node_delta_idx(0) * 1000000.0f) / 1000000.0f;
    EXPECT_EQ(roundedValue, out[0]);

    
    delete curr_lyr;
    delete mae;
}
