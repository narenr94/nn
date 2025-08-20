#include <gtest/gtest.h>
#include "poolingLayer.h"
#include "inputLayer.h"
#include "convLayer.h"


TEST(PoolingLayer_TESTS, PoolingLayer_max_setup_test)
{
    sLayer_Dimensions dims, dims2;
    dims.unInputColumns = 4;
    dims.unInputRows = 4;
    dims.unNoTransformParameterMtx = 1;
    dims.unOutputColumns = 2;
    dims.unOutputRows = 2;
    dims.unTransformParametersColumns = 2;
    dims.unTransformParametersRows = 2;

    dims2.unInputColumns = 0;
    dims2.unInputRows = 0;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 16;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 0;
    dims2.unTransformParametersRows = 0;

    PoolingLayer* lyr = new PoolingLayer(dims, ePooling_type::MAX);
    InputLayer* prev_lyr = new InputLayer(dims2, eAct_func::RELU, 0.999f);

    lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    EXPECT_EQ(lyr->get_num_nodes(), 4);
    EXPECT_EQ(lyr->get_transform_matrix_parameter_size(), 4);

    delete lyr;
}

TEST(PoolingLayer_TESTS, PoolingLayer_save_load_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 4;
    dims.unInputRows = 16;
    dims.unNoTransformParameterMtx = 4;
    dims.unOutputColumns = 2;
    dims.unOutputRows = 8;
    dims.unTransformParametersColumns = 2;
    dims.unTransformParametersRows = 2;

    PoolingLayer* lyr = new PoolingLayer(dims, ePooling_type::MAX);

    PoolingLayer* lyr2 = new PoolingLayer(lyr->get_serialized_save_data());

    EXPECT_EQ(lyr2->get_layer_dimensions().unInputColumns, dims.unInputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unInputRows, dims.unInputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unNoTransformParameterMtx, dims.unNoTransformParameterMtx);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputColumns, dims.unOutputColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unOutputRows, dims.unOutputRows);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersColumns, dims.unTransformParametersColumns);
    EXPECT_EQ(lyr2->get_layer_dimensions().unTransformParametersRows, dims.unTransformParametersRows);

    EXPECT_EQ(lyr2->get_num_nodes(), 16);

    EXPECT_EQ(lyr2->get_pooling_type(), ePooling_type::MAX);


    delete lyr;

    delete lyr2;

}