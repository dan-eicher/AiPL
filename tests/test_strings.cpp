// String operations tests
// Tests for STRING type, character vectors, and UTF-8 support

#include <gtest/gtest.h>
#include "machine.h"
#include "parser.h"
#include "value.h"

using namespace apl;

class StringTest : public ::testing::Test {
protected:
    Machine* machine = nullptr;

    void SetUp() override {
        machine = new Machine();
    }

    void TearDown() override {
        delete machine;
    }

    Value* eval(const std::string& input) {
        Value* result = machine->eval(input);
        if (!result) {
            ADD_FAILURE() << "Eval failed: " << machine->parser->get_error();
        }
        return result;
    }
};

// ============================================================================
// Basic String Literals
// ============================================================================

TEST_F(StringTest, StringLiteral) {
    Value* result = eval("'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "hello");
}

TEST_F(StringTest, EmptyString) {
    Value* result = eval("''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "");
}

TEST_F(StringTest, StringWithSpaces) {
    Value* result = eval("'hello world'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "hello world");
}

// ============================================================================
// ISO 13751 Section 6.1.5: Literal-Conversion Edge Cases
// ============================================================================

TEST_F(StringTest, SingleCharacterIsScalar) {
    // ISO 13751 6.1.5: "A single character between quotes is a scalar"
    Value* result = eval("'A'");
    ASSERT_NE(result, nullptr);
    // In our implementation, single-char strings stay as STRING type
    // but when converted to numeric (via ravel), they become scalar
    Value* raveled = eval(",'A'");
    ASSERT_NE(raveled, nullptr);
    EXPECT_TRUE(raveled->is_vector());
    EXPECT_EQ(raveled->size(), 1);
}

TEST_F(StringTest, EmbeddedQuotes) {
    // ISO 13751 6.1.5: "A quote character is represented... by two adjacent quote characters"
    // 'it''s' should become the string "it's"
    Value* result = eval("'it''s'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "it's");
}

TEST_F(StringTest, QuoteCharacterScalar) {
    // ISO 13751 6.1.5: "the character literal '''' is the character scalar 'quote'"
    Value* result = eval("''''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "'");
}

TEST_F(StringTest, MultipleEmbeddedQuotes) {
    // Multiple embedded quotes: 'say ''hi'' to me'
    Value* result = eval("'say ''hi'' to me'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "say 'hi' to me");
}

TEST_F(StringTest, ConsecutiveQuotes) {
    // Three quotes in a row: ''''''' = two quote characters
    Value* result = eval("''''''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "''");
}

TEST_F(StringTest, QuoteAtStart) {
    // Quote at start of string
    Value* result = eval("'''hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "'hello");
}

TEST_F(StringTest, QuoteAtEnd) {
    // Quote at end of string
    Value* result = eval("'hello'''");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "hello'");
}

// ============================================================================
// String Pool and GC Integration Tests
// ============================================================================

TEST_F(StringTest, StringPoolInterning) {
    // Same string should return same String* pointer
    String* s1 = machine->string_pool.intern("test");
    String* s2 = machine->string_pool.intern("test");
    EXPECT_EQ(s1, s2);  // Same pointer
}

TEST_F(StringTest, StringPoolDifferentStrings) {
    // Different strings should return different pointers
    String* s1 = machine->string_pool.intern("foo");
    String* s2 = machine->string_pool.intern("bar");
    EXPECT_NE(s1, s2);
}

TEST_F(StringTest, StringSurvivesGCWhenReachable) {
    // Assign string to variable, run GC, verify it's still accessible
    eval("x←'hello world'");

    // Force a major GC
    machine->heap->collect(machine);

    // Variable should still be accessible
    Value* result = eval("x");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "hello world");
}

TEST_F(StringTest, InternedNamesSurviveGC) {
    // Variable names are interned strings - they should survive GC
    size_t initial_size = machine->string_pool.size();

    eval("myVar←42");
    size_t after_define = machine->string_pool.size();
    EXPECT_GE(after_define, initial_size);

    // Force major GC
    machine->heap->collect(machine);

    // myVar should still be accessible
    Value* result = eval("myVar");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(StringTest, EnvironmentKeysMarkedDuringGC) {
    // Define several variables with interned names
    eval("alpha←1");
    eval("beta←2");
    eval("gamma←3");

    // Force GC
    machine->heap->collect(machine);

    // All should still be accessible
    EXPECT_DOUBLE_EQ(eval("alpha")->as_scalar(), 1.0);
    EXPECT_DOUBLE_EQ(eval("beta")->as_scalar(), 2.0);
    EXPECT_DOUBLE_EQ(eval("gamma")->as_scalar(), 3.0);
}

TEST_F(StringTest, StringValueSurvivesMultipleGCs) {
    // Assign string value and verify it survives multiple GC cycles
    eval("s←'persistent string'");

    for (int i = 0; i < 5; i++) {
        machine->heap->collect(machine);
    }

    Value* result = eval("s");
    ASSERT_NE(result, nullptr);
    EXPECT_STREQ(result->as_string()->c_str(), "persistent string");
}

TEST_F(StringTest, FunctionNamesInCacheSurviveGC) {
    // Define a function, which caches its name
    eval("f←{⍵+1}");

    // Force GC
    machine->heap->collect(machine);

    // Function should still work
    Value* result = eval("f 5");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// ============================================================================
// String to Char Vector Conversion (via primitives)
// ============================================================================

TEST_F(StringTest, ShapeOfString) {
    // Shape of string should give its length (after converting to char vector)
    Value* result = eval("⍴'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 5.0);
}

TEST_F(StringTest, TallyOfString) {
    // Tally of string should give its length
    Value* result = eval("≢'hello'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 5.0);
}

TEST_F(StringTest, RavelString) {
    // Ravel of string converts to char vector
    Value* result = eval(",'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // Should be codepoints for 'A' and 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);
}

TEST_F(StringTest, ReverseString) {
    // Reverse of string
    Value* result = eval("⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Should be C, B, A (67, 66, 65)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 67.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 65.0);
}

// ============================================================================
// String Concatenation
// ============================================================================

TEST_F(StringTest, CatenateStrings) {
    // Concatenating two strings
    Value* result = eval("'hello','world'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 10);
}

TEST_F(StringTest, CatenateStringAndVector) {
    // String with numeric vector (both become char vectors)
    Value* result = eval("'A',66");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);  // scalar 66
}

// ============================================================================
// String Comparison
// ============================================================================

TEST_F(StringTest, CompareStringElements) {
    // Compare string elements (as codepoints)
    Value* result = eval("'ABC'='ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // All should be 1 (equal)
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 1.0);
}

TEST_F(StringTest, CompareStringWithScalar) {
    // Compare string with scalar codepoint
    Value* result = eval("65='ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    // Only first element (A=65) should match
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 0.0);
}

// ============================================================================
// Execute with Strings
// ============================================================================

TEST_F(StringTest, ExecuteString) {
    Value* result = eval("⍎'1+2'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 3.0);
}

TEST_F(StringTest, ExecuteStringVariable) {
    // Assign string to variable then execute
    eval("code←'2×3'");
    Value* result = eval("⍎code");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 6.0);
}

// ============================================================================
// UTF-8 Support
// ============================================================================

TEST_F(StringTest, UTF8Greek) {
    // Greek letter alpha (2-byte UTF-8)
    Value* result = eval("⍴'α'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 1.0);  // 1 character
}

TEST_F(StringTest, UTF8APLSymbols) {
    // APL iota symbol (3-byte UTF-8)
    Value* result = eval(",'⍳'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x2373);  // U+2373
}

TEST_F(StringTest, UTF8Emoji) {
    // Emoji (4-byte UTF-8)
    Value* result = eval(",'\xF0\x9F\x98\x80'");  // U+1F600 grinning face
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 1);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x1F600);
}

TEST_F(StringTest, UTF8MixedLength) {
    // Mixed: ASCII + 2-byte + 3-byte
    Value* result = eval("⍴'Aα€'");
    ASSERT_NE(result, nullptr);
    // Should be 3 characters, not 6 bytes
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 3.0);
}

TEST_F(StringTest, UTF8EmptyString) {
    // Empty string has no codepoints
    Value* result = eval("⍴''");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0.0);
}

TEST_F(StringTest, UTF8AllASCII) {
    // Pure ASCII string
    Value* result = eval("⍴'ABCDE'");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 5.0);
}

TEST_F(StringTest, UTF8CodepointValues) {
    // Verify specific codepoint values
    Value* result = eval(",'αβγ'");  // Greek lowercase letters
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x03B1);  // α = U+03B1
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0x03B2);  // β = U+03B2
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 0x03B3);  // γ = U+03B3
}

TEST_F(StringTest, UTF8APLSymbolValues) {
    // Verify APL symbol codepoints
    Value* result = eval(",'⍳⍴⍪'");
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 0x2373);  // ⍳ = U+2373
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 0x2374);  // ⍴ = U+2374
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 0x236A);  // ⍪ = U+236A
}

// ============================================================================
// ISO 13751 Section 10.2.19-21: Dyadic Character Grade
// ============================================================================

TEST_F(StringTest, DyadicGradeUpWithCollating) {
    // ISO 13751: Dyadic ⍋ uses left arg as collating sequence
    // 'BAC' ⍋ 'ABC' → sort ABC using BAC order (B<A<C)
    Value* result = eval("'BAC'⍋'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    // In BAC order: B is first, A is second, C is third
    // 'ABC' sorted by BAC order: B(pos 2) < A(pos 1) < C(pos 3)
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);  // 'B' at index 2
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);  // 'A' at index 1
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 3.0);  // 'C' at index 3
}

TEST_F(StringTest, DyadicGradeDownWithCollating) {
    // Dyadic ⍒ uses left arg as collating sequence (descending)
    Value* result = eval("'BAC'⍒'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    // Descending BAC order: C > A > B
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 3.0);  // 'C' at index 3
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 1.0);  // 'A' at index 1
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);  // 'B' at index 2
}

TEST_F(StringTest, DyadicGradeWithDuplicates) {
    // Collating sequence with duplicates uses first occurrence
    Value* result = eval("'ABBA'⍋'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    // 'A' appears first in 'ABBA', 'B' appears second
    // So A < B, meaning 'AB' is already sorted
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 2.0);
}

TEST_F(StringTest, DyadicGradeNotInCollating) {
    // Character not in collating sequence - sorts after all others
    Value* result = eval("'AB'⍋'AXB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    // 'X' not in 'AB', so it sorts last
    // A(1) < B(3) < X(2)
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 'A'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);  // 'B'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);  // 'X' (not in collating, sorts last)
}

// ============================================================================
// Arithmetic on Strings (Character Codepoints)
// ============================================================================

TEST_F(StringTest, AddToString) {
    // Arithmetic on character data is a DOMAIN ERROR per ISO 13751
    EXPECT_THROW(eval("1+'A'"), APLError);
    EXPECT_THROW(eval("1+'ABC'"), APLError);
}

TEST_F(StringTest, SubtractStrings) {
    // Arithmetic on character data is a DOMAIN ERROR per ISO 13751
    EXPECT_THROW(eval("'B'-'A'"), APLError);
    EXPECT_THROW(eval("'ABC'-1"), APLError);
}

// ============================================================================
// String with Operators
// ============================================================================

TEST_F(StringTest, ReduceOverString) {
    // Sum of codepoints
    Value* result = eval("+/'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 131.0);  // 65 + 66
}

TEST_F(StringTest, ReduceAxisOverString) {
    // +/[1] on string should work same as +/ (strings are 1D)
    Value* result = eval("+/[1]'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 198.0);  // 65 + 66 + 67
}

TEST_F(StringTest, ScanAxisOverString) {
    // +\[1] on string gives running sum of codepoints
    Value* result = eval("+\\[1]'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // A
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 131.0);  // A+B
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 198.0);  // A+B+C
}

TEST_F(StringTest, EachOverString) {
    // Negate each codepoint
    Value* result = eval("-¨'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), -65.0);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), -66.0);
}

// ============================================================================
// String Indexing Tests (using char vector form)
// ============================================================================

TEST_F(StringTest, IndexStringScalar) {
    // S←,'ABC' ⋄ S[2] → 66 (character 'B')
    eval("S←,'ABC'");
    Value* result = eval("S[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 66.0);  // 'B' = 66
}

TEST_F(StringTest, IndexStringFirst) {
    // S←,'hello' ⋄ S[1] → 104 (character 'h')
    eval("S←,'hello'");
    Value* result = eval("S[1]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 104.0);  // 'h' = 104
}

TEST_F(StringTest, IndexStringLast) {
    // S←,'hello' ⋄ S[5] → 111 (character 'o')
    eval("S←,'hello'");
    Value* result = eval("S[5]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 111.0);  // 'o' = 111
}

TEST_F(StringTest, IndexStringVector) {
    // S←,'ABCDE' ⋄ S[2 4] → 66 68 (characters 'B' 'D')
    eval("S←,'ABCDE'");
    Value* result = eval("S[2 4]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 2);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 68.0);  // 'D'
}

// ============================================================================
// String Indexed Assignment Tests (using char vector form)
// ============================================================================

TEST_F(StringTest, IndexedAssignStringScalar) {
    // S←,'ABC' ⋄ S[2]←90 ⋄ S[2] → 90 (replace 'B' with 'Z')
    eval("S←,'ABC'");
    eval("S[2]←90");  // 'Z' = 90

    Value* result = eval("S[2]");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 90.0);
}

TEST_F(StringTest, IndexedAssignStringFirst) {
    // S←,'abc' ⋄ S[1]←65 ⋄ S → 65 98 99 (replace 'a' with 'A')
    eval("S←,'abc'");
    eval("S[1]←65");  // 'A' = 65

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 98.0);   // 'b'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 99.0);   // 'c'
}

TEST_F(StringTest, IndexedAssignStringLast) {
    // S←,'xyz' ⋄ S[3]←33 ⋄ S → 120 121 33 (replace 'z' with '!')
    eval("S←,'xyz'");
    eval("S[3]←33");  // '!' = 33

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 120.0);  // 'x'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 121.0);  // 'y'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 33.0);   // '!'
}

TEST_F(StringTest, IndexedAssignStringReturnsValue) {
    // The return value of S[I]←V is V
    eval("S←,'test'");
    Value* result = eval("S[2]←88");  // 'X' = 88
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 88.0);
}

TEST_F(StringTest, IndexedAssignStringChained) {
    // Multiple indexed assignments to build a different string
    eval("S←,'----'");
    eval("S[1]←65");   // 'A'
    eval("S[2]←66");   // 'B'
    eval("S[3]←67");   // 'C'
    eval("S[4]←68");   // 'D'

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);  // 'C'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 68.0);  // 'D'
}

TEST_F(StringTest, IndexedAssignStringDirectNoRavel) {
    // S←'ABC' ⋄ S[2]←90 → string converted to numeric, then modified
    // Without ravel, string stays STRING type until indexed assignment
    eval("S←'ABC'");
    eval("S[2]←90");  // 'Z' = 90, should convert string to numeric

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());  // Should now be numeric vector
    EXPECT_EQ(result->size(), 3);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 90.0);   // 'Z' (was 'B')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);   // 'C'
}

TEST_F(StringTest, IndexedAssignStringVectorIndices) {
    // S←'ABCDE' ⋄ S[2 4]←88 89 → modify positions 2 and 4
    eval("S←'ABCDE'");
    eval("S[2 4]←88 89");  // 'X'=88, 'Y'=89

    Value* result = eval("S");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 5);
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(0, 0), 65.0);   // 'A'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(1, 0), 88.0);   // 'X' (was 'B')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(2, 0), 67.0);   // 'C'
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(3, 0), 89.0);   // 'Y' (was 'D')
    EXPECT_DOUBLE_EQ(result->as_matrix()->coeff(4, 0), 69.0);   // 'E'
}

// ============================================================================
// Character Grading (⍋ ⍒ on strings)
// ============================================================================

TEST_F(StringTest, GradeUpString) {
    // ⍋'cab' → indices that sort to 'abc': 2 3 1 (1-origin per ISO 13751)
    Value* result = eval("⍋'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 2.0);  // 'a' at index 2
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);  // 'b' at index 3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 1.0);  // 'c' at index 1
}

TEST_F(StringTest, GradeDownString) {
    // ⍒'cab' → indices that sort descending: 1 3 2 (1-origin per ISO 13751)
    Value* result = eval("⍒'cab'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 1.0);  // 'c' at index 1
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 3.0);  // 'b' at index 3
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 2.0);  // 'a' at index 2
}

TEST_F(StringTest, GradeUpSortString) {
    // Use grade to sort: 'cab'[⍋'cab'] → 'abc'
    Value* result = eval("'cab'[⍋'cab']");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
    auto* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0, 0), 97.0);   // 'a'
    EXPECT_DOUBLE_EQ((*mat)(1, 0), 98.0);   // 'b'
    EXPECT_DOUBLE_EQ((*mat)(2, 0), 99.0);   // 'c'
}

// ============================================================================
// Format (⍕) - ISO 13751 Section 15.4
// ============================================================================

// Monadic Format - Character Passthrough
TEST_F(StringTest, FormatCharacterStringPassthrough) {
    Value* result = eval("⍕'hello'");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "hello");
}

TEST_F(StringTest, FormatEmptyStringPassthrough) {
    Value* result = eval("⍕''");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "");
}

// Monadic Format - Scalar Formatting
TEST_F(StringTest, FormatIntegerScalar) {
    Value* result = eval("⍕42");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "42");
}

TEST_F(StringTest, FormatNegativeInteger) {
    Value* result = eval("⍕¯5");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "¯5");
}

TEST_F(StringTest, FormatZero) {
    Value* result = eval("⍕0");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "0");
}

TEST_F(StringTest, FormatFloatScalar) {
    Value* result = eval("⍕3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(StringTest, FormatNegativeFloat) {
    Value* result = eval("⍕¯3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Monadic Format - Vector Formatting
TEST_F(StringTest, FormatIntegerVector) {
    Value* result = eval("⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "1 2 3");
}

TEST_F(StringTest, FormatVectorWithNegatives) {
    Value* result = eval("⍕¯1 2 ¯3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "¯1 2 ¯3");
}

// Monadic Format - Empty Arrays
TEST_F(StringTest, FormatEmptyVector) {
    Value* result = eval("⍕⍬");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    EXPECT_STREQ(result->as_string()->c_str(), "");
}

// Monadic Format - Print Precision
TEST_F(StringTest, FormatPrintPrecision3) {
    machine->pp = 3;
    Value* result = eval("⍕3.14159265");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // With pp=3, should have at most 3 significant digits
    EXPECT_TRUE(s.length() <= 6);  // "3.14" or similar
}

TEST_F(StringTest, FormatPrintPrecision10) {
    machine->pp = 10;
    Value* result = eval("⍕3.14159265");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // With pp=10, should preserve more digits
    EXPECT_TRUE(s.find("3.14159") != std::string::npos);
}

// Monadic Format - Exponential Form
TEST_F(StringTest, FormatLargeNumber) {
    Value* result = eval("⍕1e15");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Should use exponential form for large numbers
    EXPECT_TRUE(s.find("E") != std::string::npos || s.find("e") != std::string::npos);
}

TEST_F(StringTest, FormatSmallNumber) {
    Value* result = eval("⍕0.0000001");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Should use exponential form for very small numbers
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

// Monadic Format - Matrix
TEST_F(StringTest, FormatMatrix) {
    Value* result = eval("⍕2 3⍴⍳6");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Should have newline between rows
    EXPECT_TRUE(s.find("\n") != std::string::npos);
}

// Note: Infinity format tests are in test_primitives.cpp where we can directly
// allocate INFINITY values. Per ISO 13751 Section 9.2.1.1, ÷0 throws a DOMAIN ERROR
// ("domain-error is returned...such as one divided-by zero"), not infinity.

// Dyadic Format - Fixed Decimal
TEST_F(StringTest, DyadicFormatFixedBasic) {
    Value* result = eval("5 2⍕3.14159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Width 5, 2 decimal places: " 3.14"
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("3.14") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatFixedZeroDecimals) {
    Value* result = eval("5 0⍕42.7");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Width 5, 0 decimals, rounds: "   43"
    EXPECT_EQ(s.length(), 5);
    EXPECT_TRUE(s.find("43") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatFixedNegative) {
    Value* result = eval("6 2⍕¯3.14");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    EXPECT_EQ(s.length(), 6);
    EXPECT_TRUE(s.find("¯3.14") != std::string::npos);
}

// Dyadic Format - Exponential
TEST_F(StringTest, DyadicFormatExponentialBasic) {
    Value* result = eval("10 ¯3⍕3.14159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Negative precision means exponential form
    EXPECT_TRUE(s.find("E") != std::string::npos);
}

TEST_F(StringTest, DyadicFormatExponentialLarge) {
    Value* result = eval("10 ¯3⍕31415.9");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("4") != std::string::npos);  // exponent should be 4
}

TEST_F(StringTest, DyadicFormatExponentialSmall) {
    Value* result = eval("12 ¯3⍕0.00314159");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    EXPECT_TRUE(s.find("E") != std::string::npos);
    EXPECT_TRUE(s.find("¯") != std::string::npos);  // negative exponent
}

// Dyadic Format - Vector with Single Spec
TEST_F(StringTest, DyadicFormatVectorSingleSpec) {
    Value* result = eval("6 2⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_string());
    std::string s = result->as_string()->str();
    // Each element formatted with width 6
    EXPECT_EQ(s.length(), 18);  // 3 * 6
}

// Dyadic Format - Error Cases
TEST_F(StringTest, DyadicFormatWidthTooNarrow) {
    // Width too narrow for the number
    EXPECT_THROW(eval("3 2⍕123.456"), APLError);
}

TEST_F(StringTest, DyadicFormatNonNumericRight) {
    // Right argument must be numeric
    EXPECT_THROW(eval("5 2⍕'abc'"), APLError);
}

TEST_F(StringTest, DyadicFormatOddLengthSpec) {
    // Spec must have even length (pairs)
    EXPECT_THROW(eval("5 2 3⍕1 2"), APLError);
}

TEST_F(StringTest, DyadicFormatZeroWidth) {
    // Width must be positive
    EXPECT_THROW(eval("0 2⍕42"), APLError);
}

// ============================================================================
// Character Data Preservation Tests
// Verify is_char_data() flag is preserved through array operations
// ============================================================================

TEST_F(StringTest, ReshapePreservesCharData) {
    Value* result = eval("3 3 ⍴ 'abcdefghi'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReshapePreservesCharDataCyclic) {
    // Cyclic fill should still preserve char data
    Value* result = eval("2 4 ⍴ 'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReshapeToVectorPreservesCharData) {
    Value* result = eval("5 ⍴ 'abc'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RavelPreservesCharData) {
    Value* result = eval(",'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RavelMatrixPreservesCharData) {
    Value* result = eval(",2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TakePreservesCharData) {
    Value* result = eval("3↑'ABCDE'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TakeOverfillPreservesCharData) {
    // Taking more than available fills with zeros but should preserve char flag
    Value* result = eval("5↑'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, DropPreservesCharData) {
    Value* result = eval("2↓'ABCDE'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReversePreservesCharData) {
    Value* result = eval("⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RotatePreservesCharData) {
    Value* result = eval("1⌽'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, TransposePreservesCharData) {
    Value* result = eval("⍉2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, CatenatePreservesCharData) {
    Value* result = eval("'ABC','DEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, CatenateFirstPreservesCharData) {
    Value* result = eval("(2 3⍴'ABCDEF')⍪(1 3⍴'GHI')");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, FirstPreservesCharData) {
    // ↑'ABC' returns scalar, scalars don't have char flag
    // but ↑ on matrix should preserve it
    Value* result = eval("↑2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    // First of matrix is first row (a vector)
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReverseFirstPreservesCharData) {
    Value* result = eval("⊖2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, RotateFirstPreservesCharData) {
    Value* result = eval("1⊖2 3⍴'ABCDEF'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_matrix());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ReplicatePreservesCharData) {
    Value* result = eval("1 0 1/'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, ExpandPreservesCharData) {
    Value* result = eval("1 0 1 0 1\\'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, IndexingPreservesCharData) {
    Value* result = eval("'ABCDE'[2 4]");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

TEST_F(StringTest, EachPreservesCharData) {
    // ⊢¨'ABC' should return each character, preserving char flag
    Value* result = eval("⊢¨'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
}

// Round-trip tests (⍎⍕)
TEST_F(StringTest, FormatExecuteRoundTripInteger) {
    Value* result = eval("⍎⍕42");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), 42.0);
}

TEST_F(StringTest, FormatExecuteRoundTripNegative) {
    Value* result = eval("⍎⍕¯17");
    ASSERT_NE(result, nullptr);
    EXPECT_DOUBLE_EQ(result->as_scalar(), -17.0);
}

TEST_F(StringTest, FormatExecuteRoundTripFloat) {
    machine->pp = 17;  // Full precision
    Value* result = eval("⍎⍕3.14");
    ASSERT_NE(result, nullptr);
    EXPECT_NEAR(result->as_scalar(), 3.14, 1e-10);
}

TEST_F(StringTest, FormatExecuteRoundTripVector) {
    Value* result = eval("⍎⍕1 2 3");
    ASSERT_NE(result, nullptr);
    ASSERT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 3);
}

// ============================================================================
// ISO 13751: Value-Value Juxtaposition with Strings is SYNTAX ERROR
// (Lexer-level strands only work for adjacent numeric literals)
// ============================================================================

TEST_F(StringTest, MixedNumericStringJuxtapositionIsSyntaxError) {
    // ISO 13751: value-value juxtaposition is SYNTAX ERROR
    // Strings and numbers adjacent without a function between them
    EXPECT_THROW(eval("1 2 'a'"), APLError);
    EXPECT_THROW(eval("1 'ab' 2"), APLError);
    EXPECT_THROW(eval("'x' 1 2"), APLError);
    EXPECT_THROW(eval("'a' 'b' 'c'"), APLError);
    EXPECT_THROW(eval("1 'a'"), APLError);
    EXPECT_THROW(eval("'ab' 1 2"), APLError);
    EXPECT_THROW(eval("1 2 'ab'"), APLError);
}

TEST_F(StringTest, StringsWorkWithFunctions) {
    // Strings work fine when there's a function involved
    // Catenate strings: 'ab','cd' = 'abcd' (97 98 99 100)
    Value* result = eval("'ab','cd'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_EQ(result->size(), 4);
}

// ============================================================================
// Typical Element Tests (ISO 13751 §5.3.2)
// Character arrays use blank ' ' (32), numeric arrays use 0
// ============================================================================

TEST_F(StringTest, FirstOfEmptyCharArrayReturnsBlank) {
    // First (↑) of empty character array returns typical element (blank)
    Value* result = eval("↑0↑'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 32.0);  // blank = ASCII 32
}

TEST_F(StringTest, FirstOfEmptyNumericArrayReturnsZero) {
    // First (↑) of empty numeric array returns typical element (zero)
    Value* result = eval("↑0↑1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_scalar());
    EXPECT_DOUBLE_EQ(result->as_scalar(), 0.0);
}

TEST_F(StringTest, TakeExtendCharArrayWithBlanks) {
    // Taking more than length pads with blanks for character data
    Value* result = eval("5↑'AB'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ((*mat)(1), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ((*mat)(2), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(3), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(4), 32.0);  // blank
}

TEST_F(StringTest, TakeExtendNumericArrayWithZeros) {
    // Taking more than length pads with zeros for numeric data
    Value* result = eval("5↑1 2");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_FALSE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(2), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(3), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(4), 0.0);
}

TEST_F(StringTest, ExpandCharArrayWithBlanks) {
    // Expand (backslash) uses blank for character fill
    Value* result = eval("1 0 1 0 1\\'ABC'");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_TRUE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 65.0);  // 'A'
    EXPECT_DOUBLE_EQ((*mat)(1), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(2), 66.0);  // 'B'
    EXPECT_DOUBLE_EQ((*mat)(3), 32.0);  // blank
    EXPECT_DOUBLE_EQ((*mat)(4), 67.0);  // 'C'
}

TEST_F(StringTest, ExpandNumericArrayWithZeros) {
    // Expand (backslash) uses zero for numeric fill
    Value* result = eval("1 0 1 0 1\\1 2 3");
    ASSERT_NE(result, nullptr);
    EXPECT_TRUE(result->is_vector());
    EXPECT_FALSE(result->is_char_data());
    EXPECT_EQ(result->size(), 5);
    const Eigen::MatrixXd* mat = result->as_matrix();
    EXPECT_DOUBLE_EQ((*mat)(0), 1.0);
    EXPECT_DOUBLE_EQ((*mat)(1), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(2), 2.0);
    EXPECT_DOUBLE_EQ((*mat)(3), 0.0);
    EXPECT_DOUBLE_EQ((*mat)(4), 3.0);
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
