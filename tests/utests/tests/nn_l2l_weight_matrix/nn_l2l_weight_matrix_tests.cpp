#include <gtest/gtest.h>
#include "nn_l2l_weight_matrix.h"


TEST(NN_L2L_WT_MTX_TESTS, nn_l2l_weight_matrix_setup_test)
{
    nn_layer* in_lyr = new nn_layer(2, eAct_func::RELU, 0.0f);

    nn_layer* out_lyr = new nn_layer(3, eAct_func::RELU, 0.0f);

    nn_l2l_weight_matrix* wtMtx = new nn_l2l_weight_matrix(in_lyr, out_lyr);

    EXPECT_EQ(wtMtx->get_size(), (in_lyr->get_num_nodes() * out_lyr->get_num_nodes()));

    delete in_lyr;
    delete out_lyr;
    delete wtMtx;
}


TEST(NN_L2L_WT_MTX_TESTS, nn_l2l_weight_matrix_set_get_wt_test)
{
    nn_layer* in_lyr = new nn_layer(2, eAct_func::RELU, 0.0f);

    nn_layer* out_lyr = new nn_layer(3, eAct_func::RELU, 0.0f);

    nn_l2l_weight_matrix* wtMtx = new nn_l2l_weight_matrix(in_lyr, out_lyr);

    float val[6] = {0.5f, 0.11f, 0.3f, -0.2f, 0.02f};

    wtMtx->set_all_weight(val);

    for(int i = 0; i < 6; i++)
    {
        EXPECT_EQ(wtMtx->get_weight(i), val[i]);
    }

    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            EXPECT_EQ(wtMtx->get_weight(i, j), val[(i * 3) + j]);
        }
    }

    delete in_lyr;
    delete out_lyr;
    delete wtMtx;
}