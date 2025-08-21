#include "poolingLayer.h"

#include "baseAccelerator.h"
#include <cassert>

#define NOT_APPLICABLE_FOR_POOLING_LAYER assert(0)

std::map<ePooling_type, std::string> poolingTypeToString = {
        {AVERAGE, "AVERAGE"},
        {MAX, "MAX"},
        {NA, "NA"}
    };

std::map<std::string, ePooling_type> stringToPoolingType = {
        {"AVERAGE", AVERAGE},
        {"MAX", MAX},
        {"NA", NA}
    };

PoolingLayer::PoolingLayer(sLayer_Dimensions t_dims, ePooling_type t_pooling_type):BaseLayer(eLayer_type::POOLING, t_dims, eAct_func::TANH, static_cast<float>(static_cast<int>(t_pooling_type)))
{
    assert(t_pooling_type != ePooling_type::NA);
    m_pooling_type = t_pooling_type;
}

PoolingLayer::PoolingLayer(std::string load_data):BaseLayer(eLayer_type::POOLING, load_data)
{
    deserialize_load_data(load_data);
}

PoolingLayer::~PoolingLayer()
{
}

void PoolingLayer::do_forwardpass_to_current_layer()
{
    assert(m_bPrevNxtLyrsSet == true);
    m_pAccelerator->do_forwardpass_pooling_layer(m_pooling_type);
}

void PoolingLayer::set_transform_matrix_parameter(uint unInIdx, uint unOutIdx, float fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

void PoolingLayer::set_transform_matrix_parameter(uint Idx, float fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

void PoolingLayer::set_all_transform_matrix_parameter(float* fWt)
{
    NOT_APPLICABLE_FOR_POOLING_LAYER;
}

std::string PoolingLayer::get_serialized_save_data()
{
    std::ostringstream ss;
    ss << "m_pooling_type: ";
    assert(poolingTypeToString.find(m_pooling_type) != poolingTypeToString.end());
    ss << poolingTypeToString[m_pooling_type] << " \n";
    return get_serialized_general_layer_data() + ss.str();
}

void PoolingLayer::deserialize_load_data(std::string load_data)
{
    bool pooling_type_set = false;

    std::vector<std::string> all_lines = split_by_lines(load_data);

    std::vector<std::pair<std::string, std::vector<std::string>>> parsed_lines;

    for(uint i = 0; i < all_lines.size(); i++)
    {
        parsed_lines.push_back(parse_line(all_lines[i]));
    }

    for(uint i = 0; i < parsed_lines.size(); i++)
    {
        if(parsed_lines[i].first == "m_pooling_type:")
        {
            m_pooling_type = stringToPoolingType[parsed_lines[i].second[0]];
            pooling_type_set = true;
            break;
        }
    }

    assert(pooling_type_set);
}

ePooling_type PoolingLayer::get_pooling_type()
{
    return m_pooling_type;
}

