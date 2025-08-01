#include "convLayer.h"

#include "baseAccelerator.h"

#include <cassert>

ConvLayer::ConvLayer(sLayer_Dimensions t_dims, eAct_func eActFunc, float actParam1):BaseLayer(eLayer_type::CONV, t_dims, eActFunc, actParam1)
{
}

ConvLayer::ConvLayer(std::string load_data):BaseLayer(eLayer_type::CONV, load_data)
{
}

ConvLayer::~ConvLayer()
{
}

void ConvLayer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_conv_layer(m_Dimensions);
}

void ConvLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(unInIdx < (m_Dimensions.unTransformParametersRows * m_Dimensions.unNoTransformParameterMtx));
    assert(unOutIdx < m_Dimensions.unTransformParametersColumns);

    m_pfTransformParameters[(unInIdx * m_Dimensions.unTransformParametersColumns) + unOutIdx] = fWt;

}

void ConvLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    assert(Idx < m_unTransformMatrixSize);
    
    m_pfTransformParameters[Idx] = fWt;

}



void ConvLayer::set_all_transform_matrix_parameter(float* fWt)
{
    assert(m_bPrevNxtLyrsSet == true);
    uint i = 0;
    uint j = 0;
    uint f = 0;

    uint filterSz = m_Dimensions.unTransformParametersRows * m_Dimensions.unTransformParametersColumns;

    for(f = 0; f < m_Dimensions.unNoTransformParameterMtx; f++)
    {
        for(i = 0; i < m_Dimensions.unTransformParametersRows; i++)
        {
            for(j = 0; j < m_Dimensions.unTransformParametersColumns; j++)
            {
                m_pfTransformParameters[((i * m_Dimensions.unTransformParametersColumns) + j) + (f * filterSz)] = fWt[((i * m_Dimensions.unTransformParametersColumns) + j) + (f * filterSz)];
            }
        }
    }
}

std::string ConvLayer::get_serialized_save_data()
{
    return get_serialized_general_layer_data() + get_serialized_biases_data() + get_serialized_transform_mtx_data();
}

