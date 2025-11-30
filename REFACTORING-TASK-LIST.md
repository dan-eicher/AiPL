# APL-Eigen CEK Machine Refactoring Task List

## Overview
This refactoring focuses on **unifying the memory management model** to eliminate the hybrid GC/manual allocation patterns. The goal is to make all heap-allocated objects use a consistent allocation interface while preserving all existing APL functionality.

**Key Principle:** All tests must continue to pass after each task. This is a pure refactoring - no functionality changes.

---

## Phase 1: Unify GC Allocation Interface (Foundation)

### Task 1.1: Add template-based allocation to APLHeap ✅ COMPLETED
**Difficulty:** Low
**Risk:** Low
**Files:** `include/heap.h`, `src/heap.cpp`

- [x] Add template method to APLHeap
- [x] Add template specializations for Value and Continuation types
- [x] Write unit tests for new allocation interface
- [x] Run all existing tests to verify no regression

**Success Criteria:** New template allocation works, all tests pass ✅

---

### Task 1.2: Migrate Value allocations to use template interface ✅ COMPLETED
**Difficulty:** Low
**Risk:** Low
**Files:** `src/continuation.cpp`, `src/parser.cpp`, `src/primitives.cpp`

- [x] Replace pattern `heap->allocate(Value::from_scalar(x))` with `heap->allocate_scalar(x)`
- [x] Update all continuation invoke() methods
- [x] Run all tests after each file change

**Success Criteria:** All Value allocations go through heap, all tests pass ✅

---

### Task 1.3: Migrate Continuation allocations to use template interface ✅ COMPLETED
**Difficulty:** Low
**Risk:** Low
**Files:** `src/continuation.cpp`, `src/parser.cpp`

- [x] Replace pattern `new XK(...); heap->allocate_continuation(k)` with `heap->allocate<XK>(...)`
- [x] Update all continuation creation sites in continuation.cpp
- [x] Update all continuation creation sites in parser.cpp
- [x] Run tests after each major change

**Success Criteria:** No raw `new Continuation*` calls outside heap, all tests pass ✅

---

## Phase 2: Make Completions GC-Managed

### Task 2.1: Add GCObject base class ✅ COMPLETED
**Difficulty:** Low
**Risk:** Low
**Files:** `include/heap.h`, `include/value.h`, `include/continuation.h`

- [x] Create abstract GCObject base class
- [x] Make Value inherit from GCObject (remove duplicate marked/in_old_generation fields)
- [x] Make Continuation inherit from GCObject (remove duplicate fields)
- [x] Update APLHeap to manage `std::vector<GCObject*>` instead of separate vectors
- [x] Run all tests to verify inheritance works

**Success Criteria:** Value and Continuation share common GC base, all tests pass ✅

---

### Task 2.2+2.3+2.4: Make APLCompletion GC-managed, optimize NORMAL, enforce heap-only allocation ✅ COMPLETED
**Difficulty:** High
**Risk:** High
**Files:** All source and test files

**What was completed:**
- [x] Make APLCompletion inherit from GCObject
- [x] Implement APLCompletion::mark() to trace value pointer
- [x] Change Control::completion from raw pointer to GC pointer
- [x] Remove all `delete completion` calls (GC will handle it)
- [x] Update APLHeap to track completions in GC lists
- [x] Allocate completions through heap: `heap->allocate<APLCompletion>(...)`
- [x] Update all completion creation sites (ReturnK, LeaveK, etc.)
- [x] Add completion roots to Machine::mark_from_roots()
- [x] Optimize NORMAL completions to use nullptr (no allocation)
- [x] Update Machine::execute() to check `completion != nullptr`
- [x] **ENFORCE heap-only allocation via private operator new/delete**
- [x] Add private `operator new` and `operator delete` to Value, Continuation, APLCompletion
- [x] Fix all source files to use heap allocation (primitives.cpp, parser.cpp)
- [x] Fix all test files to use heap allocation (7 test files)
- [x] Remove all manual `new`/`delete` calls throughout codebase
- [x] Run all tests, especially control flow tests

**Success Criteria:** Completions are GC-managed, NORMAL uses nullptr, compiler enforces heap-only allocation, all tests pass ✅

**Note:** Combined tasks 2.2, 2.3, and jumped ahead to do enforcement (originally task 8.1) because we're on a branch and don't need backwards compatibility.

---

## Phase 3: Make Environments GC-Managed

### Task 3.1: Make Environment GC-managed ✅ COMPLETED
**Difficulty:** Medium
**Risk:** Medium
**Files:** `include/environment.h`, `src/environment.cpp`, `include/machine.h`, `src/machine.cpp`

- [x] Make Environment inherit from GCObject
- [x] Update Environment::mark() to mark parent pointer:
  ```cpp
  void mark(APLHeap* heap) override {
      for (auto& [_, val] : bindings) {
          if (val) heap->mark_value(val);
      }
      if (parent) heap->mark_object(parent);  // Mark parent environment
  }
  ```
- [x] Add APLHeap methods to track Environment objects
- [x] Change Machine::env from raw pointer to GC pointer
- [x] Remove `delete env` from Machine destructor
- [x] Update FunctionCallK and RestoreEnvK to use GC-allocated environments
- [x] Allocate environments through heap: `heap->allocate<Environment>(parent)`
- [x] Add environment roots to Machine::mark_from_roots()
- [x] Run all tests, especially function call tests

**Success Criteria:** Environments are GC-managed, no manual deletes, all tests pass ✅

---

### Task 3.2: Test environment chain GC ✅ COMPLETED
**Difficulty:** Low
**Risk:** Low
**Files:** `tests/test_heap.cpp`

- [x] Add test for environment chain marking (EnvironmentChainMarking)
- [x] Add test for deep environment nesting (DeepEnvironmentNesting - 15 levels)
- [x] Add test for environment promotion (EnvironmentPromotion - documented no generational GC)
- [x] Add test for environment survival during GC (EnvironmentSurvivesGC, EnvironmentValueLifecycle)
- [x] Add test for unreachable environment collection (UnreachableEnvironmentCollectedMajorGC, UnreachableEnvironmentCollectedViaCollect)
- [x] Run GC stress tests

**Success Criteria:** Environment GC tests pass, no leaks detected ✅

**Note:** Added 7 comprehensive environment GC tests. Unreachable environments are collected during major GC (not minor GC which only sweeps young generation Values).

---

## Phase 4: Fix Primitive Allocation (Critical Bug Fix)

### Task 4.1: Add Machine context to primitive functions
**Difficulty:** High
**Risk:** High
**Files:** `include/primitives.h`, `src/primitives.cpp`, `include/value.h`

**Strategy:** Change primitive function signatures to accept Machine pointer

- [ ] Update PrimitiveFn struct:
  ```cpp
  struct PrimitiveFn {
      const char* name;
      Value* (*monadic)(Machine* m, Value* omega);
      Value* (*dyadic)(Machine* m, Value* lhs, Value* rhs);
  };
  ```
- [ ] Update all primitive function signatures (20+ functions)
- [ ] Update all primitive implementations to use `m->heap->allocate_scalar()` etc.
- [ ] Update ApplyMonadicK::invoke() to pass machine pointer
- [ ] Update ApplyDyadicK::invoke() to pass machine pointer
- [ ] Run tests incrementally after each primitive is updated

**Success Criteria:** All primitives allocate through heap, all tests pass

---

### Task 4.2: Remove Value factory methods
**Difficulty:** Low
**Risk:** Low
**Files:** `include/value.h`, `src/value.cpp`

- [ ] Mark Value::from_scalar() as deprecated
- [ ] Mark Value::from_vector() as deprecated
- [ ] Mark Value::from_matrix() as deprecated
- [ ] Mark Value::from_primitive() as deprecated
- [ ] Mark Value::from_closure() as deprecated
- [ ] Search for any remaining usage of factory methods
- [ ] Remove factory methods entirely
- [ ] Run all tests

**Success Criteria:** No factory methods remain, all tests pass

---

## Phase 5: Parser Integration with GC

### Task 5.1: Give Parser access to heap
**Difficulty:** Low
**Risk:** Low
**Files:** `include/parser.h`, `src/parser.cpp`

- [ ] Add APLHeap* member to Parser class
- [ ] Update Parser constructor to accept heap pointer
- [ ] Replace all `new LiteralK(...)` with `heap->allocate<LiteralK>(...)`
- [ ] Replace all `new MonadicK(...)` with `heap->allocate<MonadicK>(...)`
- [ ] Replace all `new DyadicK(...)` with `heap->allocate<DyadicK>(...)`
- [ ] Update all other continuation allocations in parser
- [ ] Run parser tests after each change

**Success Criteria:** Parser allocates continuations through heap, all tests pass

---

### Task 5.2: Test parser GC integration
**Difficulty:** Low
**Risk:** Low
**Files:** `tests/test_parser.cpp`

- [ ] Add test for GC during parsing
- [ ] Add test for large expression GC (100+ continuations)
- [ ] Add test that parser doesn't leak continuations
- [ ] Run parser tests with GC stress

**Success Criteria:** Parser GC tests pass, no leaks

---

## Phase 6: String Management Unification

### Task 6.1: Unify string ownership using StringPool
**Difficulty:** Low
**Risk:** Low
**Files:** `include/continuation.h`, `src/continuation.cpp`, `src/parser.cpp`

- [ ] Replace `std::string var_name` with `const char* var_name` in LookupK
- [ ] Replace `std::string var_name` with `const char* var_name` in AssignK
- [ ] Replace `std::string var_name` with `const char* var_name` in PerformAssignK
- [ ] Replace `std::string var_name` with `const char* var_name` in ForK
- [ ] Replace `std::string var_name` with `const char* var_name` in ForIterateK
- [ ] Update all continuation constructors to accept StringPool& and intern strings
- [ ] Update Parser to pass string_pool to continuation constructors
- [ ] Run all tests

**Success Criteria:** All strings interned through StringPool, all tests pass

---

## Phase 7: Optimization (Optional)

### Task 7.1: Add continuation pooling infrastructure
**Difficulty:** Medium
**Risk:** Low
**Files:** `include/heap.h`, `src/heap.cpp`

- [ ] Add continuation pools to APLHeap:
  ```cpp
  template<typename K>
  class ContinuationPool {
      std::vector<K*> free_list;
  public:
      K* acquire();
      void release(K* k);
  };
  ```
- [ ] Add pools for common continuations (ApplyMonadicK, ApplyDyadicK, etc.)
- [ ] Update heap template allocation to check pool first
- [ ] Add continuation recycling during sweep phase
- [ ] Benchmark before/after pooling
- [ ] Run all tests

**Success Criteria:** Continuation pooling reduces allocations, all tests pass

---

## Phase 8: Cleanup and Documentation

### Task 8.1: Remove deprecated interfaces
**Difficulty:** Low
**Risk:** Low
**Files:** `include/heap.h`, `src/heap.cpp`

- [ ] Remove old `allocate(Value*)` method
- [ ] Remove old `allocate_continuation(Continuation*)` method
- [ ] Remove any remaining backward compatibility shims
- [ ] Update all comments and documentation
- [ ] Run all tests

**Success Criteria:** Clean interface, all tests pass

---

### Task 8.2: Update documentation
**Difficulty:** Low
**Risk:** None
**Files:** `README.md`, new `ARCHITECTURE.md`

- [ ] Document unified GC model
- [ ] Document allocation patterns
- [ ] Document memory safety guarantees
- [ ] Add architecture diagram showing GC relationships
- [ ] Update task list with refactoring completion notes

**Success Criteria:** Documentation complete and accurate

---

## Testing Strategy

**After EVERY task:**
1. Run `ctest` in build directory
2. All 339 tests must pass
3. No new compiler warnings
4. Run valgrind memory leak detection (if available)

**Before moving to next phase:**
1. Full test suite passes
2. Code review of changes
3. Commit with descriptive message
4. Tag with phase number (e.g., `refactor-phase-1-complete`)

---

## Rollback Plan

Each phase is independent and can be reverted:
- Phase 1: Rollback to old allocation patterns
- Phase 2: Rollback to manual completion management
- Phase 3: Rollback to manual environment management
- Phase 4: Keep old primitive signatures
- Phases 5-8: Optional enhancements

---

## Risk Mitigation

**High-Risk Tasks:**
- Task 4.1 (Primitive signatures) - Many files affected
- Task 2.2 (Completion GC) - Complex control flow
- Task 3.1 (Environment GC) - Complex lifetime

**Mitigation:**
1. Create feature branch for each high-risk task
2. Incremental commits with detailed messages
3. Run tests after every 10-20 lines of changes
4. Keep old code paths until new code is proven

---

## Success Metrics

**Before Refactoring:**
- 3 allocation models (GC Values/Continuations, manual Completions/Environments, unmanaged primitives)
- 10+ different allocation patterns
- Manual delete calls in 5+ files

**After Refactoring:**
- 1 unified allocation model (all through heap)
- 1 allocation pattern (`heap->allocate<T>(...)`)
- 0 manual delete calls (except in heap destructor)
- All 339 tests still passing
- No memory leaks detected

---

## Estimated Effort

| Phase | Tasks | Est. Time | Risk |
|-------|-------|-----------|------|
| Phase 1 | 3 tasks | 2-3 hours | Low |
| Phase 2 | 3 tasks | 3-4 hours | Medium |
| Phase 3 | 2 tasks | 2-3 hours | Medium |
| Phase 4 | 2 tasks | 4-5 hours | High |
| Phase 5 | 2 tasks | 2-3 hours | Low |
| Phase 6 | 1 task | 1-2 hours | Low |
| Phase 7 | 2 tasks | 3-4 hours | Low |
| Phase 8 | 2 tasks | 1-2 hours | None |
| **Total** | **17 tasks** | **18-26 hours** | - |

---

## Notes

- **No APL functionality changes** - This is pure internal refactoring
- **All tests must pass** at every checkpoint
- **Backward compatibility** maintained until final cleanup
- **Git commits** after each successful task
- **Can pause between phases** without breaking functionality

This refactoring will make the codebase significantly cleaner and more maintainable while preserving all existing APL semantics.
