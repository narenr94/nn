#include <gtest/gtest.h>
#include "nn_core.h"
#include <cmath>


TEST(NN_CORE_TESTS, nn_core_setup_test)
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
    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->unSzLys[l] = sz[l];
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

    EXPECT_EQ(nn->GetNumLys(), 3);
    EXPECT_EQ(nn->GetLearningRate(), 0.01f);
    for(uint i = 0; i < initData->unNoLys; i++)
    {
        EXPECT_EQ(nn->GetSzLayer(i), initData->unSzLys[i]);
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

TEST(NN_CORE_TESTS, nn_core_testrun_test)
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
    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->unSzLys[l] = sz[l];
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

TEST(NN_CORE_TESTS, nn_core_trainrun_test)
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
    for(uint l = 0; l < initData->unNoLys; l++)
    {
        initData->unSzLys[l] = sz[l];
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


