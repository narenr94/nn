#include "cpuAccelerator.h"

CpuAccelerator::CpuAccelerator(BaseLayer* pLayer):BaseAccelerator(pLayer)
{
}

CpuAccelerator::~CpuAccelerator()
{
}

void CpuAccelerator::do_forwardpass_dense_layer()
{
}

void CpuAccelerator::do_backwardpass_from_output_layer(float* pfExpOut, BaseLossFunction* lossFunc)
{
}

void CpuAccelerator::do_backwardpass_dense_layer()
{
}

void CpuAccelerator::do_forwardpass_conv_layer(sLayer_Dimensions t_tims)
{
}

void CpuAccelerator::do_backwardpass_conv_layer(sLayer_Dimensions t_tims)
{
}

void CpuAccelerator::do_forwardpass_pooling_layer(ePooling_type t_pooling_type)
{
}

void CpuAccelerator::do_backwardpass_pooling_layer(ePooling_type t_pooling_type)
{
}