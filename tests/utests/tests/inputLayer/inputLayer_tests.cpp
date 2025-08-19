#include <gtest/gtest.h>
#include "inputLayer.h"


TEST(InputLayer_TESTS, InputLayer_setup_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 0;
    dims.unInputRows = 0;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 0;
    dims.unTransformParametersRows = 0;

    InputLayer* lyr = new InputLayer(dims, eAct_func::RELU, 0.999f);

    EXPECT_EQ(lyr->get_num_nodes(), 5);
	EXPECT_EQ(lyr->get_act_func(), eAct_func::RELU);
	EXPECT_EQ(lyr->get_act_param(), 0.999f);

    delete lyr;
}


TEST(InputLayer_TESTS, InputLayer_set_get_node_data_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 0;
    dims.unInputRows = 0;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 0;
    dims.unTransformParametersRows = 0;

    float in[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    InputLayer* lyr = new InputLayer(dims, eAct_func::RELU, 0.999f);

    for(uint i = 0; i < 5; i++)
    {
        lyr->set_node_value(in[i], i);
    }

    for(uint i = 0; i < 5; i++)
    {
        EXPECT_EQ(lyr->get_node_value_idx(i), in[i]);
    }    

    delete lyr;

}


TEST(InputLayer_TESTS, InputLayer_save_load_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 0;
    dims.unInputRows = 0;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 0;
    dims.unTransformParametersRows = 0;

    InputLayer* lyr = new InputLayer(dims, eAct_func::RELU, 0.999f);

    InputLayer* lyr2 = new InputLayer(lyr->get_serialized_save_data());

    EXPECT_EQ(lyr2->get_num_nodes(), 5);

    EXPECT_EQ(lyr2->get_layer_dimensions().unInputColumns, dims.unInputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unInputRows, dims.unInputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unNoTransformParameterMtx, dims.unNoTransformParameterMtx);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputColumns, dims.unOutputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputRows, dims.unOutputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersColumns, dims.unTransformParametersColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersRows, dims.unTransformParametersRows);

    delete lyr;

    delete lyr2;

}
