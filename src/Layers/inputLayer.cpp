#include "inputLayer.h"

#include "baseAccelerator.h"
#include <cassert>

#define NOT_APPLICABLE_FOR_INPUT_LAYER assert(0)

InputLayer::InputLayer(sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1):BaseLayer(eLayer_type::INPUT, t_dims, eActFunc, actParam1)
{
}

InputLayer::~InputLayer()
{
}

void InputLayer::SetPreviousNextLayers(BaseLayer* prevLyr, BaseLayer* nxtLyr)
{
    assert(prevLyr == nullptr);
    m_pPrevLyr = prevLyr;
    m_pNextLyr = nxtLyr;

    m_unTransformMatrixSize = 0;
    m_bPrevNxtLyrsSet = true;
}

void InputLayer::do_forwardpass_to_current_layer()
{
    NOT_APPLICABLE_FOR_INPUT_LAYER;
}

void InputLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    NOT_APPLICABLE_FOR_INPUT_LAYER;
}

void InputLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    NOT_APPLICABLE_FOR_INPUT_LAYER;
}

void InputLayer::set_all_transform_matrix_parameter(float* fWt)
{
    NOT_APPLICABLE_FOR_INPUT_LAYER;
}