#include <gtest/gtest.h>
#include "nn_math.h"
#include <cmath>


TEST(NN_MATH_TESTS, nn_math_sigmoid)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.622459f, 0.683521f, 0.554779f, 0.377541f, 0.445221f, 0.0f, 1.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(get_sigmoidf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_sigmoidf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.25f, 0.1771f, 0.1716f, -0.75f, -0.2684f, -54522.0f, -261632.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_sigmoidf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_randomNumber)
{
    float min[7] = {7, 2, 15, 12, 0, 6, 15};
    float max[7] = {10, 5, 30, 20, 5, 10, 45};

    for(int i = 0; i < 7; i++)
    {
        uint num = getRandomNumber(min[i], max[i]);
        EXPECT_GE(num, min[i]);
        EXPECT_LE(num, max[i]);
    }
    
}

TEST(NN_MATH_TESTS, nn_math_relu)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.5f, 0.77f, 0.22f, 0.0f, 0.0f, 0.0f, 512.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(get_reluf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_reluf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_reluf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_leakyReluf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.5f, 0.77f, 0.22f, -0.005f, -0.0022f, -2.33f, 512.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(get_leakyReluf(in[i], 0.01f) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_leakyReluf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {1.0f, 1.0f, 1.0f, 0.01f, 0.01f, 0.01f, 1.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_leakyReluf(in[i], 0.01f) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_tanhf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.462117f, 0.646929f, 0.216518f, -0.462117f, -0.216518f, -1.0f, 1.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(get_tanhf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_tanhf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.75f, 0.4071f, 0.9516f, 0.75f, 0.9516f, -54288.0f, -262143.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_tanhf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_softmaxf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {1.0f, 0.852144f, 0.251579f, 0.332871f, 0.895834f, 0.0f, 1.0f};
    float max[7] = {0.5f, 0.93f, 1.6f, 0.6f, -0.11f, 0.5f, 512.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(get_softmaxf(in[i], max[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_softmaxf)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {0.25f, 0.1771f, 0.1716f, -0.75f, -0.2684f, -54522.0f, -261632.0f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_softmaxf(in[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

TEST(NN_MATH_TESTS, nn_math_derivative_softmaxf_wrong_pred)
{
    float in[7] = {0.5f, 0.77f, 0.22f, -0.5f, -0.22f, -233.0f, 512.0f};
    float out[7] = {-0.3f, -0.7161f, -0.352f, 0.3f, -0.0242f, 116.5f, -1075.2f};
    float corr[7] = {0.6f, 0.93f, 1.6f, 0.6f, -0.11f, 0.5f, 2.1f};

    for(int i = 0; i < 7; i++)
    {
        float roundedValue = std::round(find_derivative_softmaxf_wrong_pred(in[i], corr[i]) * 1000000.0f) / 1000000.0f;
        EXPECT_EQ(roundedValue, out[i]);
    }
}

