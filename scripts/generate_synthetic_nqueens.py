#!/usr/bin/env python3
"""
Generate synthetic N-Queens training traces without using an API.

Since N-Queens has a well-known backtracking solution, we can generate
high-quality training examples programmatically.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from axiom.procedural.generators_hard import NQueensCountGenerator


# Canonical N-Queens solution with variations
NQUEENS_SOLUTIONS = [
    # Solution 1: Classic backtracking with column/diagonal sets
    {
        "thinking": """Step 1: Understand the problem - N-Queens asks to count all valid placements of N queens on an NxN chessboard where no two queens attack each other.

Step 2: Key insight - Queens attack on rows, columns, and diagonals. We can place exactly one queen per row, so we iterate row by row.

Step 3: Use backtracking with constraint tracking:
- Track occupied columns with a set
- Track occupied diagonals (row-col) with a set
- Track occupied anti-diagonals (row+col) with a set

Step 4: Time complexity O(N!), Space O(N) for the recursion stack and sets.

Step 5: Base case - when row == n, we found a valid placement, count it.""",
        "code": '''def n_queens(n: int) -> int:
    count = 0
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col

    def backtrack(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            backtrack(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return count'''
    },
    # Solution 2: Using lists instead of sets
    {
        "thinking": """Step 1: N-Queens problem - count valid queen placements where no two queens share row, column, or diagonal.

Step 2: Approach - place one queen per row using backtracking. Track constraints with boolean arrays.

Step 3: For each row, try each column. Check if column and both diagonals are free.

Step 4: Diagonal constraint: queens on same diagonal have same (row-col), same anti-diagonal have same (row+col).

Step 5: Use arrays of size n for columns, 2n-1 for diagonals.""",
        "code": '''def n_queens(n: int) -> int:
    result = 0
    col_used = [False] * n
    diag_used = [False] * (2 * n - 1)
    anti_diag_used = [False] * (2 * n - 1)

    def solve(row):
        nonlocal result
        if row == n:
            result += 1
            return
        for col in range(n):
            d1 = row - col + n - 1
            d2 = row + col
            if col_used[col] or diag_used[d1] or anti_diag_used[d2]:
                continue
            col_used[col] = diag_used[d1] = anti_diag_used[d2] = True
            solve(row + 1)
            col_used[col] = diag_used[d1] = anti_diag_used[d2] = False

    solve(0)
    return result'''
    },
    # Solution 3: Compact version with helper function
    {
        "thinking": """Step 1: The N-Queens counting problem requires finding all ways to place N non-attacking queens.

Step 2: Two queens attack if they share row, column, or diagonal. Row constraint is automatic (one per row).

Step 3: Use backtracking - at each row, try placing queen in each column that doesn't conflict.

Step 4: Check conflicts using: columns set, positive diagonal (r-c), negative diagonal (r+c).

Step 5: When we successfully place queen in last row (row==n), we found one valid configuration.""",
        "code": '''def n_queens(n: int) -> int:
    def is_safe(queens, row, col):
        for r, c in enumerate(queens):
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def backtrack(queens):
        row = len(queens)
        if row == n:
            return 1
        count = 0
        for col in range(n):
            if is_safe(queens, row, col):
                count += backtrack(queens + [col])
        return count

    return backtrack([])'''
    },
    # Solution 4: Bit manipulation (advanced)
    {
        "thinking": """Step 1: N-Queens - count arrangements of N queens on NxN board with no attacks.

Step 2: Use bit manipulation for efficiency. Each bit represents a column position.

Step 3: Track three constraints as bitmasks: columns, left diagonals, right diagonals.

Step 4: Available positions = ~(cols | diag_left | diag_right) & ((1 << n) - 1)

Step 5: Extract lowest set bit with pos & -pos, iterate through all available positions.""",
        "code": '''def n_queens(n: int) -> int:
    def solve(row, cols, d1, d2):
        if row == n:
            return 1
        count = 0
        available = ~(cols | d1 | d2) & ((1 << n) - 1)
        while available:
            pos = available & -available
            available -= pos
            count += solve(row + 1, cols | pos, (d1 | pos) << 1, (d2 | pos) >> 1)
        return count

    return solve(0, 0, 0, 0)'''
    },
    # Solution 5: Iterative version
    {
        "thinking": """Step 1: N-Queens asks for the count of valid placements on an NxN board.

Step 2: Track queen positions in a list where queens[i] = column of queen in row i.

Step 3: For conflict checking: same column means queens[i] == queens[j], diagonal means |i-j| == |queens[i]-queens[j]|.

Step 4: Use recursive backtracking, placing one queen per row.

Step 5: Return 1 when we've placed n queens (reached row n), sum all valid continuations.""",
        "code": '''def n_queens(n: int) -> int:
    def can_place(queens, row, col):
        for i in range(row):
            if queens[i] == col or abs(queens[i] - col) == row - i:
                return False
        return True

    count = 0
    queens = [-1] * n

    def place(row):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if can_place(queens, row, col):
                queens[row] = col
                place(row + 1)
                queens[row] = -1

    place(0)
    return count'''
    },
    # Solution 6: Clean recursive with early termination
    {
        "thinking": """Step 1: N-Queens counting problem - find number of ways to place N queens safely.

Step 2: Safe placement means no two queens in same row, column, or diagonal.

Step 3: Process row by row. For each row, try all columns. Skip invalid positions.

Step 4: Track used columns and diagonals. Diagonals identified by row-col (positive) and row+col (negative).

Step 5: Increment counter when row equals n (all queens placed successfully).""",
        "code": '''def n_queens(n: int) -> int:
    def solve(row, cols, d1, d2):
        if row == n:
            return 1
        total = 0
        for col in range(n):
            if col not in cols and (row - col) not in d1 and (row + col) not in d2:
                total += solve(row + 1, cols | {col}, d1 | {row - col}, d2 | {row + col})
        return total

    return solve(0, set(), set(), set())'''
    },
    # Solution 7: Named constraints
    {
        "thinking": """Step 1: N-Queens problem requires counting non-attacking queen arrangements on NxN board.

Step 2: Queens attack horizontally, vertically, and diagonally. Place one queen per row.

Step 3: For each position (row, col), check: column not used, main diagonal (row-col) not used, anti-diagonal (row+col) not used.

Step 4: Use backtracking: try each column, mark as used, recurse, then unmark.

Step 5: Base case returns 1 (found valid solution), sum all possibilities.""",
        "code": '''def n_queens(n: int) -> int:
    used_cols = set()
    used_main_diag = set()
    used_anti_diag = set()

    def count_solutions(row):
        if row == n:
            return 1
        solutions = 0
        for col in range(n):
            main_diag = row - col
            anti_diag = row + col
            if col in used_cols or main_diag in used_main_diag or anti_diag in used_anti_diag:
                continue
            used_cols.add(col)
            used_main_diag.add(main_diag)
            used_anti_diag.add(anti_diag)
            solutions += count_solutions(row + 1)
            used_cols.remove(col)
            used_main_diag.remove(main_diag)
            used_anti_diag.remove(anti_diag)
        return solutions

    return count_solutions(0)'''
    },
    # Solution 8: Alternative diagonal indexing
    {
        "thinking": """Step 1: Count valid N-Queens configurations where no queens attack each other.

Step 2: Key observation - place exactly one queen per row, track column and diagonal conflicts.

Step 3: For NxN board, there are n columns, 2n-1 main diagonals, 2n-1 anti-diagonals.

Step 4: Main diagonal index: col - row + n - 1 (shifted to be non-negative)
Anti-diagonal index: col + row

Step 5: Recursively try each column, backtrack on conflicts, count successful placements.""",
        "code": '''def n_queens(n: int) -> int:
    col_mask = [False] * n
    main_diag = [False] * (2 * n - 1)
    anti_diag = [False] * (2 * n - 1)

    def backtrack(row):
        if row == n:
            return 1
        count = 0
        for col in range(n):
            md = col - row + n - 1
            ad = col + row
            if col_mask[col] or main_diag[md] or anti_diag[ad]:
                continue
            col_mask[col] = main_diag[md] = anti_diag[ad] = True
            count += backtrack(row + 1)
            col_mask[col] = main_diag[md] = anti_diag[ad] = False
        return count

    return backtrack(0)'''
    },
]


def generate_synthetic_traces(output_path: Path, num_copies: int = 2):
    """Generate synthetic N-Queens traces."""

    generator = NQueensCountGenerator(seed=42)
    traces = []

    print("Generating synthetic N-Queens traces...")
    print(f"Solutions available: {len(NQUEENS_SOLUTIONS)}")
    print(f"Copies per solution: {num_copies}")

    # Generate traces for each solution variant
    for idx, solution in enumerate(NQUEENS_SOLUTIONS):
        for copy in range(num_copies):
            # Generate problem at varying difficulty
            difficulty = 3 + (idx % 5)  # Vary difficulty 3-7
            problem = generator.generate(difficulty=difficulty, num_test_cases=5)

            # Verify the solution works
            namespace = {}
            try:
                exec(solution["code"], namespace)
                func = namespace.get("n_queens")
                if func is None:
                    print(f"  Warning: Solution {idx} failed to define function")
                    continue

                # Test all cases
                passed = 0
                for tc in problem.test_cases:
                    try:
                        result = func(*tc.input_args)
                        if result == tc.expected_output:
                            passed += 1
                    except Exception as e:
                        pass

                if passed != len(problem.test_cases):
                    print(f"  Warning: Solution {idx} passed {passed}/{len(problem.test_cases)} tests")
                    continue

            except Exception as e:
                print(f"  Warning: Solution {idx} failed: {e}")
                continue

            # Build trace
            trace = {
                "problem_type": "n_queens",
                "problem_id": problem.problem_id,
                "problem_title": problem.title,
                "problem_description": problem.description,
                "function_signature": problem.function_signature,
                "function_name": problem.function_name,
                "test_cases": [
                    {"input_args": tc.input_args, "expected_output": tc.expected_output}
                    for tc in problem.test_cases
                ],
                "difficulty": difficulty,
                "thinking": solution["thinking"],
                "solution_code": solution["code"],
                "raw_response": f"<think>\n{solution['thinking']}\n</think>\n\n```python\n{solution['code']}\n```",
                "teacher_model": "synthetic",
                "verified": True,
                "passed_tests": len(problem.test_cases),
                "total_tests": len(problem.test_cases),
                "timestamp": datetime.now().isoformat(),
            }
            traces.append(trace)
            print(f"  Generated trace {len(traces)}: variant {idx}, difficulty {difficulty}")

    # Save traces
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"\nGenerated {len(traces)} verified N-Queens traces")
    print(f"Saved to: {output_path}")

    return traces


def main():
    output_path = Path("data/coldstart_v2/n_queens_synthetic.jsonl")
    traces = generate_synthetic_traces(output_path, num_copies=2)
    return 0 if traces else 1


if __name__ == "__main__":
    sys.exit(main())
