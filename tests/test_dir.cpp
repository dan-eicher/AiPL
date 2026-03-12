// Tests for DIR (Definition-site Instantiation with Re-optimization)
//
// Tests verify that dfn bodies are re-optimized at call time with
// concrete argument types, enabling patterns like E3 (EigenReduceK)
// to fire inside dfns where ⍵'s type was unknown at definition time.

#include <gtest/gtest.h>
#include "machine.h"
#include "heap.h"
#include "continuation.h"
#include "optimizer.h"
#include "dir.h"
#include <Eigen/Dense>
#include <cmath>

using namespace apl;

// ---------------------------------------------------------------------------
// Test fixture — fresh Machine per test
// ---------------------------------------------------------------------------

class DIRTest : public ::testing::Test {
protected:
    Machine* m;

    void SetUp() override {
        m = new Machine();
    }

    void TearDown() override {
        delete m;
    }

    Value* eval(const char* src) {
        return m->eval(src);
    }

    double scalar(const char* src) {
        Value* v = eval(src);
        EXPECT_NE(v, nullptr);
        EXPECT_EQ(v->tag, ValueType::SCALAR);
        return v ? v->data.scalar : 0.0;
    }
};

// ---------------------------------------------------------------------------
// 1. EigenReduceK fires inside dfn via DIR
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_EigenReduceInDfn) {
    // {+/⍵} ⍳1000 — body +/⍵ cannot become EigenReduceK at definition time
    // because ⍵ is TM_TOP. DIR re-optimizes with ⍵=TM_VECTOR → EigenReduceK.
    EXPECT_DOUBLE_EQ(scalar("{+/⍵} ⍳1000"), 500500.0);
}

TEST_F(DIRTest, DIR_ProdReduceInDfn) {
    // {×/⍵} 1 2 3 4 → 24
    EXPECT_DOUBLE_EQ(scalar("{×/⍵} 1 2 3 4"), 24.0);
}

TEST_F(DIRTest, DIR_MaxReduceInDfn) {
    // {⌈/⍵} 3 1 4 1 5 → 5
    EXPECT_DOUBLE_EQ(scalar("{⌈/⍵} 3 1 4 1 5"), 5.0);
}

TEST_F(DIRTest, DIR_MinReduceInDfn) {
    // {⌊/⍵} 3 1 4 1 5 → 1
    EXPECT_DOUBLE_EQ(scalar("{⌊/⍵} 3 1 4 1 5"), 1.0);
}

// ---------------------------------------------------------------------------
// 2. Cache behavior
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_CacheHit) {
    // Call same dfn twice with VECTOR args — second call should hit cache.
    // We verify correctness (cache hit returns correct specialized body).
    eval("f←{+/⍵}");
    EXPECT_DOUBLE_EQ(scalar("f ⍳100"), 5050.0);
    EXPECT_DOUBLE_EQ(scalar("f ⍳200"), 20100.0);  // Cache hit
}

TEST_F(DIRTest, DIR_DifferentTypes) {
    // Call with VECTOR then SCALAR — two different specializations.
    eval("f←{⍵+1}");
    Value* v1 = eval("f ⍳3");  // VECTOR arg
    EXPECT_EQ(v1->tag, ValueType::VECTOR);
    EXPECT_DOUBLE_EQ(scalar("f 5"), 6.0);  // SCALAR arg — different specialization
}

// ---------------------------------------------------------------------------
// 3. Monadic vs dyadic
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_MonadicVsDyadic) {
    // Same dfn called monadically and dyadically — different TypeSigs.
    eval("f←{⍺+⍵}");
    // Dyadic: 10 f 20
    EXPECT_DOUBLE_EQ(scalar("10 f 20"), 30.0);
    // Different dyadic types
    Value* v = eval("1 2 3 f 4 5 6");
    EXPECT_EQ(v->tag, ValueType::VECTOR);
}

// ---------------------------------------------------------------------------
// 4. Nested dfns
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_NestedDfn) {
    // Inner dfn should also get specialized via DIR.
    EXPECT_DOUBLE_EQ(scalar("{({+/⍵}⍵)} 1 2 3"), 6.0);
}

// ---------------------------------------------------------------------------
// 5. Niladic dfn — no specialization (no ⍵ argument)
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_NiladicDfn) {
    // Niladic dfns are auto-invoked — DIR shouldn't interfere.
    EXPECT_DOUBLE_EQ(scalar("{1+2}"), 3.0);
}

// ---------------------------------------------------------------------------
// 6. Inner product inside dfn (E1 pattern)
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_InnerProduct) {
    // {⍵+.×⍵} on a matrix — E1 should fire with matrix specialization.
    Value* v = eval("{⍵+.×⍵} 2 2⍴⍳4");
    EXPECT_NE(v, nullptr);
    EXPECT_EQ(v->tag, ValueType::MATRIX);
    // (1 2 / 3 4) +.× (1 2 / 3 4) = (7 10 / 15 22)
    EXPECT_DOUBLE_EQ((*v->as_matrix())(0, 0), 7.0);
    EXPECT_DOUBLE_EQ((*v->as_matrix())(0, 1), 10.0);
    EXPECT_DOUBLE_EQ((*v->as_matrix())(1, 0), 15.0);
    EXPECT_DOUBLE_EQ((*v->as_matrix())(1, 1), 22.0);
}

// ---------------------------------------------------------------------------
// 7. GC survival — specialized cache survives garbage collection
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_GCSurvival) {
    eval("f←{+/⍵}");
    EXPECT_DOUBLE_EQ(scalar("f ⍳100"), 5050.0);

    // Force GC by allocating lots of values
    for (int i = 0; i < 100; i++) {
        eval("⍳1000");
    }

    // Cache should survive — second call still correct
    EXPECT_DOUBLE_EQ(scalar("f ⍳100"), 5050.0);
}

// ---------------------------------------------------------------------------
// 8. DIR disabled when optimizer is off
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_DisabledWithOptimizer) {
    m->optimizer_enabled = false;
    // Should still produce correct results, just without specialization.
    EXPECT_DOUBLE_EQ(scalar("{+/⍵} ⍳100"), 5050.0);
    m->optimizer_enabled = true;
}

// ---------------------------------------------------------------------------
// 9. Clone correctness — complex body
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_CloneComplexBody) {
    // Dfn with control flow — clone must handle IfK, SeqK, etc.
    EXPECT_DOUBLE_EQ(scalar("{:If ⍵>0\n+/⍳⍵\n:Else\n0\n:EndIf} 10"), 55.0);
    EXPECT_DOUBLE_EQ(scalar("{:If ⍵>0\n+/⍳⍵\n:Else\n0\n:EndIf} 0"), 0.0);
}

// ---------------------------------------------------------------------------
// 10. Multiple expressions reuse same closure
// ---------------------------------------------------------------------------

TEST_F(DIRTest, DIR_PersistAcrossExpressions) {
    // f←{+/⍵} is assigned, then called in separate eval() calls.
    // The ClosureData persists; specialization cache should be reused.
    eval("f←{+/⍵}");
    EXPECT_DOUBLE_EQ(scalar("f ⍳10"), 55.0);
    EXPECT_DOUBLE_EQ(scalar("f ⍳20"), 210.0);
    EXPECT_DOUBLE_EQ(scalar("f ⍳30"), 465.0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
