#include <gtest/gtest.h>
#include "nn_defines.h"
#include <vector>

TEST(NN_DEFINES_TESTS, nn_defines_create_input_layer)
{
    std::vector<float> optParam = {1.0f, 2.0f, 3.0f};
    sMtx_Dim in_dim;
    in_dim.rows = 4;
    in_dim.columns = 4;
    nnInitData *initData = new nnInitData(in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::HUBER, 0.111f, 0.321f);

    EXPECT_EQ(initData->unNoLys, 1);
    EXPECT_EQ(initData->layer_dimensions[0].unInputRows, 0);
    EXPECT_EQ(initData->layer_dimensions[0].unInputColumns, 0);
    EXPECT_EQ(initData->layer_dimensions[0].unOutputRows, 4);
    EXPECT_EQ(initData->layer_dimensions[0].unOutputColumns, 4);
    EXPECT_EQ(initData->layer_dimensions[0].unTransformParametersRows, 0);
    EXPECT_EQ(initData->layer_dimensions[0].unTransformParametersColumns, 0);
    EXPECT_EQ(initData->layer_dimensions[0].unNoTransformParameterMtx, 1);

    EXPECT_EQ(initData->eAct_Funcs[0], eAct_func::TANH);
    EXPECT_EQ(initData->e_layer_type[0], eLayer_type::INPUT);
    EXPECT_EQ(initData->actParam1[0], 0.0f);
    EXPECT_EQ(initData->ePoolingType[0], ePooling_type::NA);

    EXPECT_EQ(initData->fLearningRate, 0.321f);
    EXPECT_EQ(initData->eOpt, eOptimizers::RMSPROP);
    EXPECT_EQ(initData->optParam[0], optParam[0]);
    EXPECT_EQ(initData->optParam[1], optParam[1]);
    EXPECT_EQ(initData->optParam[2], optParam[2]);

    EXPECT_EQ(initData->eLossFunc, eLossFuncs::HUBER);
    EXPECT_EQ(initData->lossParam, 0.111f);
}

TEST(NN_DEFINES_TESTS, nn_defines_add_dense_layer)
{
    std::vector<float> optParam = {1.0f, 2.0f, 3.0f};
    sMtx_Dim in_dim;
    in_dim.rows = 4;
    in_dim.columns = 4;
    nnInitData *initData = new nnInitData(in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::HUBER, 0.111f, 0.321f);
    initData->add_dense_layer(10, eAct_func::LEAKY_RELU, 0.312f);

    EXPECT_EQ(initData->unNoLys, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unInputRows, 1);
    EXPECT_EQ(initData->layer_dimensions[1].unInputColumns, 16);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputRows, 1);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputColumns, 10);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersRows, 16);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersColumns, 10);
    EXPECT_EQ(initData->layer_dimensions[1].unNoTransformParameterMtx, 1);

    EXPECT_EQ(initData->eAct_Funcs[1], eAct_func::LEAKY_RELU);
    EXPECT_EQ(initData->e_layer_type[1], eLayer_type::DENSE);
    EXPECT_EQ(initData->actParam1[1], 0.312f);
    EXPECT_EQ(initData->ePoolingType[1], ePooling_type::NA);

}

TEST(NN_DEFINES_TESTS, nn_defines_add_conv_layer)
{
    std::vector<float> optParam = {1.0f, 2.0f, 3.0f};

    sMtx_Dim conv_in_dim;
    conv_in_dim.rows = 4;
    conv_in_dim.columns = 4;
    nnInitData *initData = new nnInitData(conv_in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::HUBER, 0.111f, 0.321f);
    
    initData->add_conv_layer(conv_in_dim, eConvKernelSize::Sz3x3, 3, eAct_func::LEAKY_RELU, 0.312f);

    EXPECT_EQ(initData->unNoLys, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unInputRows, 4);
    EXPECT_EQ(initData->layer_dimensions[1].unInputColumns, 4);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputRows, 6);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputColumns, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersRows, 3);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersColumns, 3);
    EXPECT_EQ(initData->layer_dimensions[1].unNoTransformParameterMtx, 3);

    EXPECT_EQ(initData->eAct_Funcs[1], eAct_func::LEAKY_RELU);
    EXPECT_EQ(initData->e_layer_type[1], eLayer_type::CONV);
    EXPECT_EQ(initData->actParam1[1], 0.312f);
    EXPECT_EQ(initData->ePoolingType[1], ePooling_type::NA);

}

TEST(NN_DEFINES_TESTS, nn_defines_add_pooling_layer_single_input_mtx)
{
    std::vector<float> optParam = {1.0f, 2.0f, 3.0f};
    sMtx_Dim pooling_in_dim;
    pooling_in_dim.rows = 4;
    pooling_in_dim.columns = 4;
    nnInitData *initData = new nnInitData(pooling_in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::HUBER, 0.111f, 0.321f);
    
    initData->add_pooling_layer(pooling_in_dim, ePooling_type::MAX, ePoolingKernelSize::KrSz2x2);

    EXPECT_EQ(initData->unNoLys, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unInputRows, 4);
    EXPECT_EQ(initData->layer_dimensions[1].unInputColumns, 4);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputRows, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unOutputColumns, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersRows, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unTransformParametersColumns, 2);
    EXPECT_EQ(initData->layer_dimensions[1].unNoTransformParameterMtx, 1);

    EXPECT_EQ(initData->e_layer_type[1], eLayer_type::POOLING);
    EXPECT_EQ(initData->ePoolingType[1], ePooling_type::MAX);

}

TEST(NN_DEFINES_TESTS, nn_defines_add_pooling_layer_multiple_input_mtx)
{
    std::vector<float> optParam = {1.0f, 2.0f, 3.0f};
    sMtx_Dim conv_in_dim;
    conv_in_dim.rows = 10;
    conv_in_dim.columns = 10;
    nnInitData *initData = new nnInitData(conv_in_dim, eOptimizers::RMSPROP, optParam, eLossFuncs::HUBER, 0.111f, 0.321f);
    
    initData->add_conv_layer(conv_in_dim, eConvKernelSize::Sz3x3, 3, eAct_func::LEAKY_RELU, 0.312f);
    sMtx_Dim pooling_in_dim;
    pooling_in_dim.rows = 24;
    pooling_in_dim.columns = 8;
    initData->add_pooling_layer(pooling_in_dim, ePooling_type::AVERAGE, ePoolingKernelSize::KrSz2x2);

    EXPECT_EQ(initData->unNoLys, 3);
    EXPECT_EQ(initData->layer_dimensions[2].unInputRows, 24);
    EXPECT_EQ(initData->layer_dimensions[2].unInputColumns, 8);
    EXPECT_EQ(initData->layer_dimensions[2].unOutputRows, 12);
    EXPECT_EQ(initData->layer_dimensions[2].unOutputColumns, 4);
    EXPECT_EQ(initData->layer_dimensions[2].unTransformParametersRows, 2);
    EXPECT_EQ(initData->layer_dimensions[2].unTransformParametersColumns, 2);
    EXPECT_EQ(initData->layer_dimensions[2].unNoTransformParameterMtx, 3);

    EXPECT_EQ(initData->e_layer_type[2], eLayer_type::POOLING);
    EXPECT_EQ(initData->ePoolingType[2], ePooling_type::AVERAGE);

}

