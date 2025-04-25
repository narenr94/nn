#include <gtest/gtest.h>
#include "cpuAccelerator.h"
#include "nn_layer.h"
#include "nn_l2l_weight_matrix.h"
#include <cmath>


TEST(CPU_ACC_TESTS, cpuAccelerator_forwardpass_test)
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
