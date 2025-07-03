#include <gtest/gtest.h>
#include "nn_core.h"
#include <cmath>


TEST(NN_CORE_TESTS, nn_core_setup_dense_test)
{
    float init_lyr1_biases[3] = {0.2f, 0.5f, 0.1f};
    float init_lyr2_biases[2] = {0.6f, 0.05f};

    float init_lyr1_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float init_lyr2_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    float in[2] = {0.0f, 1.0f};
    float out[2] = {0.0f, 1.0f};

    uint sz[4] = {2,3,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

    //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 2;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 2;
    initData->layer_dimensions[1].unInputRows = 1;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 3;
    initData->layer_dimensions[1].unOutputRows = 1;
    initData->layer_dimensions[1].unTransformParametersColumns = 3;
    initData->layer_dimensions[1].unTransformParametersRows = 2;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 3;
    initData->layer_dimensions[2].unInputRows = 1;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 3;

    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->eAct_Funcs[l] = eAct_func::TANH;
        initData->e_layer_type[l] = eLayer_type::DENSE;
    }
    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    EXPECT_EQ(nn->GetNumLys(), 3);
    EXPECT_EQ(nn->GetLearningRate(), 0.01f);
    for(uint i = 0; i < initData->unNoLys; i++)
    {
        EXPECT_EQ(nn->GetSzLayer(i), initData->layer_dimensions[i].unOutputColumns * initData->layer_dimensions[i].unOutputRows);
    }

    for(uint i = 0; i < 2; i++)
    {
        EXPECT_EQ(nn->GetBias(2, i), init_lyr2_biases[i]);
    }

    for(uint i = 0; i < 3; i++)
    {
        EXPECT_EQ(nn->GetBias(1, i), init_lyr1_biases[i]);
    }

    for(uint i = 0; i < 6; i++)
    {
        EXPECT_EQ(nn->GetWeight(2, i), init_lyr2_wts[i]);
    }

    for(uint i = 0; i < 6; i++)
    {
        EXPECT_EQ(nn->GetWeight(1, i), init_lyr1_wts[i]);
    }


    delete nn;

}

TEST(NN_CORE_TESTS, nn_core_testrun_dense_test)
{
    float init_lyr1_biases[3] = {0.2f, 0.5f, 0.1f};
    float init_lyr2_biases[2] = {0.6f, 0.05f};

    float init_lyr1_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float init_lyr2_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    float in[2] = {0.0f, 1.0f};
    float out[2] = {0.302242f, 0.163498f};

    uint sz[4] = {2,3,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

    //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 2;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 2;
    initData->layer_dimensions[1].unInputRows = 1;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 3;
    initData->layer_dimensions[1].unOutputRows = 1;
    initData->layer_dimensions[1].unTransformParametersColumns = 3;
    initData->layer_dimensions[1].unTransformParametersRows = 2;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 3;
    initData->layer_dimensions[2].unInputRows = 1;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 3;

    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->e_layer_type[l] = eLayer_type::DENSE;
        initData->eAct_Funcs[l] = eAct_func::TANH;
    }
    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    nn->Test(in, out);

    for(uint i = 0; i < 2; i++)
    {
        float roundedValue = std::round(nn->GetNodeVal(2, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete nn;
}

TEST(NN_CORE_TESTS, nn_core_trainrun_dense_test)
{
    float init_lyr1_biases[3] = {0.2f, 0.5f, 0.1f};
    float init_lyr2_biases[2] = {0.6f, 0.05f};

    float init_lyr1_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    float init_lyr2_wts[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    float in[2] = {0.0f, 1.0f};
    float out[2] = {0.0f, 1.0f};

    float bias_out2[2] = {0.597254f, 0.058141f};
    float bias_out1[3] = {0.201239f, 0.501913f, 0.103114f};

    float out_lyr1_wts[6] = {0.1f, 0.2f, 0.3f, 0.401239f, 0.501913f, 0.603114f};
    float out_lyr2_wts[6] = {0.0992f, 0.202372f, 0.298731f, 0.403762f, 0.499076f, 0.602739f};

    uint sz[4] = {2,3,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

        //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 2;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 2;
    initData->layer_dimensions[1].unInputRows = 1;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 3;
    initData->layer_dimensions[1].unOutputRows = 1;
    initData->layer_dimensions[1].unTransformParametersColumns = 3;
    initData->layer_dimensions[1].unTransformParametersRows = 2;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 3;
    initData->layer_dimensions[2].unInputRows = 1;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 3;

    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->e_layer_type[l] = eLayer_type::DENSE;
        initData->eAct_Funcs[l] = eAct_func::TANH;
    }
    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    nn->Train(in, out);

    printf("outlyr delta");
    for(uint i = 0; i < 2; i++)
    {
        float roundedValue = std::round(nn->GetBias(2, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, bias_out2[i]);
        printf("%f ", nn->GetDelta(2, i));
    }
    printf("\n");

    printf("hiddenlyr delta");
    for(uint i = 0; i < 3; i++)
    {
        float roundedValue = std::round(nn->GetBias(1, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, bias_out1[i]);
        printf("%f ", nn->GetDelta(1, i));
    }
    printf("\n");

    for(uint i = 0; i < 6; i++)
    {
        float roundedValue = std::round(nn->GetWeight(2, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out_lyr2_wts[i]);
    }
    for(uint i = 0; i < 6; i++)
    {
        float roundedValue = std::round(nn->GetWeight(1, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out_lyr1_wts[i]);
    }

    delete nn;
}


TEST(NN_CORE_TESTS, nn_core_setup_conv_test)
{
    float init_lyr1_biases[9] = {0.2f, 0.5f, 0.1f,0.2f, 0.5f, 0.1f,0.2f, 0.5f, 0.1f};
    float init_lyr2_biases[2] = {0.6f, 0.05f};

    float init_lyr1_wts[16] = {0.1f, 0.2f, 0.3f, 0.4f,
                                0.1f, 0.2f, 0.3f, 0.4f,
                                0.1f, 0.2f, 0.3f, 0.4f,
                                0.1f, 0.2f, 0.3f, 0.4f};
    float init_lyr2_wts[18] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    float in[36] = {0.0f, 1.0f,0.0f, 1.0f,0.0f, 1.0f,
                    0.0f, 1.0f,0.0f, 1.0f,0.0f, 1.0f,
                    0.0f, 1.0f,0.0f, 1.0f,0.0f, 1.0f};
    float out[2] = {0.0f, 1.0f};

    uint sz[4] = {36,9,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

    //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 36;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 6;
    initData->layer_dimensions[1].unInputRows = 6;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 3;
    initData->layer_dimensions[1].unOutputRows = 3;
    initData->layer_dimensions[1].unTransformParametersColumns = 4;
    initData->layer_dimensions[1].unTransformParametersRows = 4;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 3;
    initData->layer_dimensions[2].unInputRows = 3;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 9;

    initData->eAct_Funcs[0] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[0] = eLayer_type::DENSE;

    initData->eAct_Funcs[1] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[1] = eLayer_type::CONV;

    initData->eAct_Funcs[2] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[2] = eLayer_type::DENSE;

    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    EXPECT_EQ(nn->GetNumLys(), 3);
    EXPECT_EQ(nn->GetLearningRate(), 0.01f);
    for(uint i = 0; i < initData->unNoLys; i++)
    {
        EXPECT_EQ(nn->GetSzLayer(i), initData->layer_dimensions[i].unOutputColumns * initData->layer_dimensions[i].unOutputRows);
    }

    for(uint i = 0; i < 2; i++)
    {
        EXPECT_EQ(nn->GetBias(2, i), init_lyr2_biases[i]);
    }

    for(uint i = 0; i < 3; i++)
    {
        EXPECT_EQ(nn->GetBias(1, i), init_lyr1_biases[i]);
    }

    for(uint i = 0; i < 18; i++)
    {
        EXPECT_EQ(nn->GetWeight(2, i), init_lyr2_wts[i]);
    }

    for(uint i = 0; i < 16; i++)
    {
        EXPECT_EQ(nn->GetWeight(1, i), init_lyr1_wts[i]);
    }


    delete nn;

}

TEST(NN_CORE_TESTS, nn_core_testrun_conv_test)
{
    float init_lyr1_biases[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    float init_lyr2_biases[2] = {0.2f, 0.2f};

    float init_lyr1_wts[4] = {0.2f, 0.2f,
                                0.2f, 0.2f};

    float init_lyr2_wts[8] = {0.1f, 0.2f,
                                0.3f, 0.4f,
                                0.3f, 0.4f,
                                0.1f, 0.2f};

    float in[9] = {0.2f, -0.1f, 0.6f,
                    -0.3f, 0.4f, 0.1f,
                    0.01f, 0.7f, -0.01f};

    float out[2] = {0.063525f, 0.070025f};

    uint sz[4] = {9,4,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

    //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 9;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 3;
    initData->layer_dimensions[1].unInputRows = 3;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 2;
    initData->layer_dimensions[1].unOutputRows = 2;
    initData->layer_dimensions[1].unTransformParametersColumns = 2;
    initData->layer_dimensions[1].unTransformParametersRows = 2;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 2;
    initData->layer_dimensions[2].unInputRows = 2;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 4;

    initData->eAct_Funcs[0] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[0] = eLayer_type::DENSE;

    initData->eAct_Funcs[1] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[1] = eLayer_type::CONV;

    initData->eAct_Funcs[2] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[2] = eLayer_type::DENSE;

    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::HUBER;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    nn->Test(in, out);

    for(uint i = 0; i < 2; i++)
    {
        float roundedValue = std::round(nn->GetNodeVal(2, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }

    delete nn;
}

TEST(NN_CORE_TESTS, nn_core_trainrun_conv_test)
{
    float init_lyr1_biases[4] = {0.1f, 0.1f, 0.1f, 0.1f};
    float init_lyr2_biases[2] = {0.2f, 0.2f};

    float init_lyr1_wts[4] = {0.2f, 0.2f,
                                0.2f, 0.2f};

    float init_lyr2_wts[8] = {0.1f, 0.2f,
                                0.3f, 0.4f,
                                0.3f, 0.4f,
                                0.1f, 0.2f};

    float in[9] = {0.2f, -0.1f, 0.6f,
                    -0.3f, 0.4f, 0.1f,
                    0.01f, 0.7f, -0.01f};

    float out[2] = {0.01f, 1.0f};

    float bias_out2[2] = {0.0f, 0.0f};
    float delta_out2[2] = {0.0f, 0.0f};
    float delta_out1[4] = {0.0f, 0.0f};
    float delta_out0[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};


    float bias_out1[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float out_lyr1_wts[16] = {0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f};

    float out_lyr2_wts[18] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    uint sz[4] = {9,4,2};
    nnInitData * initData = new nnInitData(3); 
    initData->unNoLys = 3;

    //input layer
    initData->layer_dimensions[0].unInputColumns = 0;
    initData->layer_dimensions[0].unInputRows = 0;
    initData->layer_dimensions[0].unNoTransformParameterMtx = 0;
    initData->layer_dimensions[0].unOutputColumns = 9;
    initData->layer_dimensions[0].unOutputRows = 1;
    initData->layer_dimensions[0].unTransformParametersColumns = 0;
    initData->layer_dimensions[0].unTransformParametersRows = 0;

    //hidden layer
    initData->layer_dimensions[1].unInputColumns = 3;
    initData->layer_dimensions[1].unInputRows = 3;
    initData->layer_dimensions[1].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[1].unOutputColumns = 2;
    initData->layer_dimensions[1].unOutputRows = 2;
    initData->layer_dimensions[1].unTransformParametersColumns = 2;
    initData->layer_dimensions[1].unTransformParametersRows = 2;

    //output layer
    initData->layer_dimensions[2].unInputColumns = 2;
    initData->layer_dimensions[2].unInputRows = 2;
    initData->layer_dimensions[2].unNoTransformParameterMtx = 1;
    initData->layer_dimensions[2].unOutputColumns = 2;
    initData->layer_dimensions[2].unOutputRows = 1;
    initData->layer_dimensions[2].unTransformParametersColumns = 2;
    initData->layer_dimensions[2].unTransformParametersRows = 4;

    initData->eAct_Funcs[0] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[0] = eLayer_type::DENSE;

    initData->eAct_Funcs[1] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[1] = eLayer_type::CONV;

    initData->eAct_Funcs[2] = eAct_func::LEAKY_RELU;
    initData->e_layer_type[2] = eLayer_type::DENSE;

    initData->eOpt = eOptimizers::SGD;
    initData->eLossFunc = eLossFuncs::MSE;
    initData->fLearningRate = 0.01f;

    NeuralNet* nn = new NeuralNet(initData);

    nn->populate_nodes_bias(1, init_lyr1_biases);
    nn->populate_nodes_bias(2, init_lyr2_biases);

    nn->populate_weights(1, init_lyr1_wts);
    nn->populate_weights(2, init_lyr2_wts);

    nn->Train(in, out);

    printf("outlyr delta");

    for(uint i = 0; i < 2; i++)
    {
        float roundedValue = std::round(nn->GetDelta(2, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, delta_out2[i]);
    }

    for(uint i = 0; i < 4; i++)
    {
        float roundedValue = std::round(nn->GetDelta(1, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, delta_out1[i]);
    }

    for(uint i = 0; i < 9; i++)
    {
        float roundedValue = std::round(nn->GetDelta(0, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, delta_out0[i]);
    }


    // for(uint i = 0; i < 2; i++)
    // {
    //     float roundedValue = std::round(nn->GetBias(2, i) * 1000000.0f) / 1000000.0f;
    //     EXPECT_EQ(roundedValue, bias_out2[i]);
    // }
    
    // for(uint i = 0; i < 4; i++)
    // {
    //     float roundedValue = std::round(nn->GetBias(1, i) * 1000000.0f) / 1000000.0f;
    //     EXPECT_EQ(roundedValue, bias_out1[i]);
    // }

    // for(uint i = 0; i < 8; i++)
    // {
    //     float roundedValue = std::round(nn->GetWeight(2, i) * 1000000.0f) / 1000000.0f;
    //     EXPECT_EQ(roundedValue, out_lyr2_wts[i]);
    // }
    for(uint i = 0; i < 4; i++)
    {
        float roundedValue = std::round(nn->GetWeight(1, i) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out_lyr1_wts[i]);
    }

    delete nn;
}


