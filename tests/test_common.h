// Common test infrastructure for APL test suite
// Provides shared fixture base class

#pragma once

#include <gtest/gtest.h>
#include "machine.h"
#include "heap.h"
#include "value.h"
#include "primitives.h"
#include "operators.h"
#include "continuation.h"
#include "parser.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace apl {
namespace test {

// Base fixture class for APL tests
// Provides Machine setup/teardown
// Tests should use machine->eval() directly for evaluation
class APLTestBase : public ::testing::Test {
protected:
    Machine* machine = nullptr;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
        machine = nullptr;
    }

    // Convenience allocators - use these or machine->heap->allocate_* directly
    Value* scalar(double v) {
        return machine->heap->allocate_scalar(v);
    }

    Value* vector(std::initializer_list<double> vals) {
        Eigen::VectorXd v(vals.size());
        int i = 0;
        for (double val : vals) {
            v(i++) = val;
        }
        return machine->heap->allocate_vector(v);
    }

    Value* matrix(int rows, int cols, std::initializer_list<double> vals) {
        Eigen::MatrixXd m(rows, cols);
        auto it = vals.begin();
        for (int i = 0; i < rows && it != vals.end(); i++) {
            for (int j = 0; j < cols && it != vals.end(); j++) {
                m(i, j) = *it++;
            }
        }
        return machine->heap->allocate_matrix(m);
    }
};

} // namespace test
} // namespace apl
