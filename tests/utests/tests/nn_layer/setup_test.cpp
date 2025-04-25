#include <gtest/gtest.h>
#include "nn_layer.h"


TEST(NN_LAYER_TESTS, nn_layer_setup_test)
{
    nn_layer* lyr = new nn_layer(5, eAct_func::RELU, 0.999f);

    EXPECT_EQ(lyr->get_num_nodes(), 5);
	EXPECT_EQ(lyr->get_act_func(), eAct_func::RELU);
	EXPECT_EQ(lyr->get_act_param(), 0.999f);

    delete lyr;

}