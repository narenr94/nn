#include "nn_l2l_weight_matrix.h"

nn_l2l_weight_matrix::nn_l2l_weight_matrix(nn_layer* pInL, nn_layer* pOutL)
{
}

nn_l2l_weight_matrix::~nn_l2l_weight_matrix()
{
}

bool nn_l2l_weight_matrix::set_weight(uint unInIdx, uint unOutIdx, float fWt)
{
    return false;
}

bool nn_l2l_weight_matrix::set_weight(uint Idx, float fWt)
{
    return false;
}

bool nn_l2l_weight_matrix::set_all_weight(float* fWt)
{
    return false;
}

float nn_l2l_weight_matrix::get_weight(uint unInIdx, uint unOutIdx)
{
    return 0.0f;
}

float nn_l2l_weight_matrix::get_weight(uint Idx)
{
    return 0.0f;
}

uint nn_l2l_weight_matrix::get_size()
{
    return 0;
}

void nn_l2l_weight_matrix::populateWeightsWithRandomNumbers()
{
}

const float* nn_l2l_weight_matrix::getWtMtx()
{
    return nullptr;
}