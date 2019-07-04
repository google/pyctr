# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Eight Queens implementation implemented in various frontends."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import z3


def z3_queens():
  """Z3 implementation from https://ericpony.github.io/z3py-tutorial/guide-examples.htm."""
  queens = [z3.Int('Q_%i' % (q + 1)) for q in range(8)]
  single_queen_per_column = [z3.Distinct(queens)]
  queen_in_column = [z3.And(1 <= queens[i], queens[i] <= 8) for i in range(8)]
  diagonal_constraint = [
      z3.And(queens[i] - queens[j] != i - j, queens[i] - queens[j] != j - i)
      for i in range(8)
      for j in range(i)
  ]

  return queen_in_column + single_queen_per_column + diagonal_constraint


def z3_python():
  """Implementation of eight queens using idiomatic Python."""
  queens = [z3.Int('Q_%i' % (q + 1)) for q in range(8)]
  single_queen_per_column = [z3.Distinct(queens)]
  queen_in_column = [(1 <= queens[i] and queens[i] <= 8) for i in range(8)]
  diagonal_constraint = []
  for i in range(8):
    for j in range(i):
      diagonal_constraint.append((queens[i] - queens[j] != i - j) and
                                 (queens[i] - queens[j] != j - i))

  return queen_in_column + single_queen_per_column + diagonal_constraint


def simplified():
  """Implementation of eight queens which uses abs.

  Note: while this appears optimized compared to previous implementations, the
  generated IR is actually more complex due to abs returning a z3.If node.

  Returns:
    Constraints for eight queens problem.
  """
  queens = [z3.Int('Q_%i' % (q + 1)) for q in range(8)]
  single_queen_per_column = [z3.Distinct(queens)]
  queen_in_column = [(1 <= queens[i] and queens[i] <= 8) for i in range(8)]
  diagonal_constraint = []
  for i in range(8):
    for j in range(i):
      diagonal_constraint.append(abs(queens[i] - queens[j]) != abs(i - j))

  return queen_in_column + single_queen_per_column + diagonal_constraint
