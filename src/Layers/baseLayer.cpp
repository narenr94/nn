#include "baseLayer.h"
#include "baseAccelerator.h"


void BaseLayer::do_backwardpass_to_previous_layer()
{
    switch(m_pNextLyr->get_layer_type())
    {
        case eLayer_type::CONV :
            m_pAccelerator->do_backwardpass_conv_layer(m_Dimensions.unInputRows, m_Dimensions.unInputColumns, m_Dimensions.unTransformParametersRows, m_Dimensions.unTransformParametersColumns);
            break;
        case eLayer_type::DENSE :
        default :
            m_pAccelerator->do_backwardpass_dense_layer();
    }
}

sLayer_Dimensions BaseLayer::get_layer_dimensions()
{
    return m_Dimensions;
}