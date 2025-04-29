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

void CpuAccelerator::do_backwardpass_dense_layer_output_layer(float* pfExpOut, BaseLossFunction* lossFunc)
{
}

void CpuAccelerator::do_backwardpass_dense_layer()
{
}