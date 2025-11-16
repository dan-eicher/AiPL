// Sanity test to verify test harness is working

#include <gtest/gtest.h>

// Simple test to verify GTest is working
TEST(SanityTest, BasicAssertion) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_TRUE(true);
    EXPECT_FALSE(false);
}

TEST(SanityTest, StringComparison) {
    std::string hello = "Hello, APL!";
    EXPECT_EQ(hello, "Hello, APL!");
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
