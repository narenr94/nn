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

InputLayer::InputLayer(std::string load_data):BaseLayer(eLayer_type::INPUT, load_data)
{

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

std::string InputLayer::get_serialized_save_data()
{
    return get_serialized_general_layer_data();
}