#include <gtest/gtest.h>
#include "convLayer.h"
#include "inputLayer.h"


TEST(ConvLayer_TESTS, ConvLayer_setup_test)
{
    sLayer_Dimensions dims, dims2;
    dims.unInputColumns = 4;
    dims.unInputRows = 4;
    dims.unNoTransformParameterMtx = 3;
    dims.unOutputColumns = 3;
    dims.unOutputRows = 9;
    dims.unTransformParametersColumns = 2;
    dims.unTransformParametersRows = 2;

    dims2.unInputColumns = 0;
    dims2.unInputRows = 0;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 16;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 0;
    dims2.unTransformParametersRows = 0;

    ConvLayer* lyr = new ConvLayer(dims, eAct_func::RELU, 0.999f);
    InputLayer* prev_lyr = new InputLayer(dims2, eAct_func::RELU, 0.999f);

    lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    EXPECT_EQ(lyr->get_num_nodes(), 27);
    EXPECT_EQ(lyr->get_act_func(), eAct_func::RELU);
    EXPECT_EQ(lyr->get_act_param(), 0.999f);
    EXPECT_EQ(lyr->get_transform_matrix_parameter_size(), 12);

    delete lyr;
}


TEST(ConvLayer_TESTS, ConvLayer_set_get_node_data_test)
{
    sLayer_Dimensions dims, dims2;
    dims.unInputColumns = 4;
    dims.unInputRows = 4;
    dims.unNoTransformParameterMtx = 3;
    dims.unOutputColumns = 3;
    dims.unOutputRows = 9;
    dims.unTransformParametersColumns = 2;
    dims.unTransformParametersRows = 6;

    dims2.unInputColumns = 0;
    dims2.unInputRows = 0;
    dims2.unNoTransformParameterMtx = 1;
    dims2.unOutputColumns = 16;
    dims2.unOutputRows = 1;
    dims2.unTransformParametersColumns = 0;
    dims2.unTransformParametersRows = 0;

    float bias[27] = {0.1f, 0.2f, 0.3f,
                    0.4f, 0.01f, 0.02f,
                    0.03f, 0.04f, 0.001f,
                    0.002f, 0.003f, 0.004f,
                    -0.1f, -0.2f, -0.3f,
                    -0.4f, 0.91f, 0.82f,
                    0.73f, 0.64f, 0.55f,
                    -0.001f, -0.002f, -0.003f,
                    -0.004f, -0.005f, 0.006f};

    float kr[12] = {0.11f, 0.22f,
                    0.33f, 0.44f,
                    0.55f, 0.66f,
                    0.77f, 0.88f,
                    -0.11f, -0.22f,
                    -0.33f, -0.44f};
                
    float val[27] = {-0.1f, -0.2f, -0.3f,
                        -0.4f, -0.5f, -0.6f,
                        0.12f, 0.23f, 0.34f,
                        0.45f, 0.56f, 0.67f,
                        -0.12f, -0.23f, -0.34f,
                        -0.45f, -0.56f, -0.67f,
                        0.4f, 0.5f, 0.6f,
                        0.1f, 0.2f, 0.3f,
                        0.111f, 0.222f, 0.333f};

    float del[27] = {0.1f, 0.2f, 0.3f,
                        0.4f, 0.5f, 0.6f,
                        -0.12f, -0.23f, -0.34f,
                        -0.45f, -0.56f, -0.67f,
                        0.12f, 0.23f, 0.34f,
                        0.45f, 0.56f, 0.67f,
                        -0.4f, -0.5f, -0.6f,
                        -0.1f, -0.2f, -0.3f,
                        -0.111f, -0.222f, -0.333f};

    ConvLayer* lyr = new ConvLayer(dims, eAct_func::RELU, 0.999f);
    InputLayer* prev_lyr = new InputLayer(dims2, eAct_func::RELU, 0.999f);

    lyr->SetPreviousNextLayers(prev_lyr, nullptr);

    for(uint i = 0; i < 27; i++)
    {
        lyr->set_node_value(val[i], i);
        lyr->set_node_bias(bias[i], i);
        lyr->set_node_delta(del[i], i);
    }

    for(uint i = 0; i < 12; i++)
    {
        lyr->set_transform_matrix_parameter(i, kr[i]);
    }

    for(uint i = 0; i < 27; i++)
    {
        EXPECT_EQ(lyr->get_node_value_idx(i), val[i]);
        EXPECT_EQ(lyr->get_node_bias_idx(i), bias[i]);
        EXPECT_EQ(lyr->get_node_delta_idx(i), del[i]);
    }

    for(uint i = 0; i < 12; i++)
    {
        EXPECT_EQ(lyr->get_transform_matrix_parameter(i), kr[i]);
    }

    delete lyr;

}

TEST(ConvLayer_TESTS, ConvLayer_save_load_test)
{
    sLayer_Dimensions dims;
    dims.unInputColumns = 4;
    dims.unInputRows = 1;
    dims.unNoTransformParameterMtx = 2;
    dims.unOutputColumns = 5;
    dims.unOutputRows = 1;
    dims.unTransformParametersColumns = 5;
    dims.unTransformParametersRows = 4;

    ConvLayer* lyr = new ConvLayer(dims, eAct_func::RELU, 0.999f);

    float biases[5] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};

    for(uint i = 0; i < 5; i++)
    {
        lyr->set_node_bias(biases[i], i);
    }

    ConvLayer* lyr2 = new ConvLayer(lyr->get_serialized_save_data());

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
