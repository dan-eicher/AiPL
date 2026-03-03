#include "machine.h"
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using namespace apl;
using Clock = std::chrono::steady_clock;

// Build "+/1 2 3 ... n"  (C1 can fold the entire sum at compile time)
static std::string lit_sum(int n) {
    std::string s = "+/";
    for (int i = 1; i <= n; i++) {
        if (i > 1) s += ' ';
        s += std::to_string(i);
    }
    return s;
}

// Build "{" + body + "}¨ ⍳N"
static std::string each(const std::string& body, int n) {
    return "{" + body + "}¨ ⍳" + std::to_string(n);
}

// Time a single eval() call on a fresh Machine.
// Run reps times; return the minimum (best-case, avoids GC noise).
static double time_script(bool opt_on, const std::string& expr, int reps = 5) {
    double best = 1e18;
    for (int i = 0; i < reps; i++) {
        Machine m;
        m.optimizer_enabled = opt_on;
        auto t0 = Clock::now();
        m.eval(expr);
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    // One eval() per case — APL does the looping, not C++.
    // Expressions chosen so execution dominates parse time.
    struct Case { const char* label; std::string expr; };

    std::vector<Case> cases = {
        // --- Controls: runtime data, optimizer cannot fold.
        //     Any difference here is pure optimizer overhead on code it can't improve.
        {"ctrl  +/⍳10000",
             "+/⍳10000"},

        // --- C1: literal vector reduce — folded to a constant at compile time.
        //     Without opt: VM executes N-1 additions at runtime.
        //     With opt:    constant is pre-computed; VM returns it instantly.
        {"C1    +/ 1000 literals",
             lit_sum(1000)},
        {"C1    +/ 10000 literals",
             lit_sum(10000)},

        // --- C1 inside ¨: dfn body has only literals — optimizer folds the
        //     body once; ¨ calls it N times.  Each call saves 7 CEK steps.
        {"C1¨   {+/lits}¨⍳1000",
             each("+/1 2 3 4 5 6 7 8", 1000)},

        // --- C2 inside ¨: constant arithmetic chain in dfn body.
        //     Without opt: ⍵+(1+(2+3)) — 3 additions per call.
        //     With opt:    ⍵+6         — 1 addition per call.
        {"C2¨   {⍵+1+2+3}¨⍳10000",
             each("⍵+1+2+3", 10000)},

        // --- O2: dfn containing a derived operator.
        //     Without opt: +/ reconstructed as DerivedOperatorK on each call.
        //     With opt:    +/ pre-built once at parse time (O2).
        //     Each element of ⍳N is a scalar so +/scalar == scalar;
        //     difference is purely the DerivedOperatorK setup cost per call.
        {"O2¨   {+/⍵}¨⍳10000",
             each("+/⍵", 10000)},

        // --- Eigen fast-path benchmarks (pure runtime, optimizer irrelevant).
        //     These test the direct Eigen dispatch bypass, not the static optimizer.

        // +/⍳N: reduce a vector — Eigen fast-path calls mat->sum() directly.
        //   Without fast-path: N-1 CEK continuation steps.
        //   With fast-path:    single Eigen sum() call.
        {"EP    +/⍳1000000",
             "+/⍳1000000"},

        // Matrix multiply: (N×N ⍴ ⍳N²) +.× (N×N ⍴ ⍳N²)
        //   Without fast-path: N² CellIterK/ReduceResultK steps.
        //   With fast-path:    single BLAS-backed Eigen matmul.
        {"EP    200×200 +.× matmul",
             "(200 200 ⍴ ⍳40000) +.× (200 200 ⍴ ⍳40000)"},
    };

    printf("%-32s  %9s  %9s  %8s\n",
           "Case", "No-Opt ms", "Opt ms", "Speedup");
    printf("%-32s  %9s  %9s  %8s\n",
           "----", "---------", "------", "-------");

    for (auto& c : cases) {
        double t_off = time_script(false, c.expr);
        double t_on  = time_script(true,  c.expr);
        printf("%-32s  %9.1f  %9.1f  %7.2fx\n",
               c.label, t_off, t_on, t_off / t_on);
    }
    return 0;
}
