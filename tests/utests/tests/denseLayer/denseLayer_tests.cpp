#include <gtest/gtest.h>
#include "denseLayer.h"


TEST(DenseLayer_TESTS, DenseLayer_setup_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 4;
    dims.unInputRows = 1;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 5;
    dims.unTransformParametersRows = 4;

    DenseLayer* lyr = new DenseLayer(dims, eAct_func::RELU, 0.999f);

    EXPECT_EQ(lyr->get_num_nodes(), 5);
	EXPECT_EQ(lyr->get_act_func(), eAct_func::RELU);
	EXPECT_EQ(lyr->get_act_param(), 0.999f);

    delete lyr;
}


TEST(DenseLayer_TESTS, DenseLayer_set_get_node_data_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 4;
    dims.unInputRows = 1;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 5;
    dims.unTransformParametersRows = 4;

    DenseLayer* lyr = new DenseLayer(dims, eAct_func::RELU, 0.999f);

    lyr->set_node_value(6.1f, 2);
    lyr->set_node_bias(3.2f, 3);
    lyr->set_node_delta(0.001f, 1);

    EXPECT_EQ(lyr->get_node_value_idx(2), 6.1f);
	EXPECT_EQ(lyr->get_node_bias_idx(3), 3.2f);
	EXPECT_EQ(lyr->get_node_delta_idx(1), 0.001f);

    delete lyr;

}

TEST(DenseLayer_TESTS, DenseLayer_save_load_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 4;
    dims.unInputRows = 1;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 5;
    dims.unTransformParametersRows = 4;

    DenseLayer* lyr = new DenseLayer(dims, eAct_func::RELU, 0.999f);

    float biases[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    for(uint i = 0; i < 5; i++)
    {
        lyr->set_node_bias(biases[i], i);
    }

    DenseLayer* lyr2 = new DenseLayer(lyr->get_serialized_save_data());

    EXPECT_EQ(lyr2->get_layer_dimensions().unInputColumns, dims.unInputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unInputRows, dims.unInputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unNoTransformParameterMtx, dims.unNoTransformParameterMtx);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputColumns, dims.unOutputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputRows, dims.unOutputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersColumns, dims.unTransformParametersColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersRows, dims.unTransformParametersRows);

    EXPECT_EQ(lyr2->get_num_nodes(), 5);

    for(uint i = 0; i < 5; i++)
    {
        EXPECT_EQ(lyr2->get_node_bias_idx(i), biases[i]);
    }

    delete lyr;

    delete lyr2;

}
