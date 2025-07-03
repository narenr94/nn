#include "baseLayer.h"
#include "baseAccelerator.h"


void BaseLayer::do_backwardpass_to_previous_layer()
{
    sLayer_Dimensions dims = m_pNextLyr->get_layer_dimensions();
    
    switch(m_pNextLyr->get_layer_type())
    {
        case eLayer_type::CONV :
            m_pAccelerator->do_backwardpass_conv_layer(dims.unInputRows, dims.unInputColumns, dims.unTransformParametersRows, dims.unTransformParametersColumns);
            break;
        case eLayer_type::DENSE :
            m_pAccelerator->do_backwardpass_dense_layer();
        default :
            m_pAccelerator->do_backwardpass_dense_layer();
    }
}

sLayer_Dimensions BaseLayer::get_layer_dimensions()
{
    return m_Dimensions;
}