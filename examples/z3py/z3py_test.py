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
"""Tests for z3py overload module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest as test
from absl.testing import parameterized
from pyctr.api import conversion
from pyctr.examples.z3py import z3py
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import logical_ops
from pyctr.transformers.virtualization import variables
import z3


def prove(f):
  s = z3.Solver()
  s.add(z3.Not(f))
  return s.check() == z3.unsat


def can_solve(f):
  s = z3.Solver()
  s.add(f)
  return s.check() == z3.sat


class Z3PyTest(parameterized.TestCase):

  def convert(self, f):
    return conversion.convert(f, z3py, [logical_ops])

  @parameterized.named_parameters(
      ('TT', True, True),
      ('TF', True, False),
      ('Tq', True, z3.Bool('q')),
      ('FT', False, True),
      ('FF', False, False),
      ('Fq', False, z3.Bool('q')),
      ('pT', z3.Bool('p'), True),
      ('pF', z3.Bool('p'), True),
      ('pq', z3.Bool('p'), z3.Bool('q')),
  )
  def test_demorgan(self, p, q):

    def demorgan(a, b):
      return (a and b) == (not (not a or not b))

    converted_demorgan = self.convert(demorgan)
    self.assertTrue(prove(converted_demorgan(p, q)))

  @parameterized.named_parameters(
      ('TTT', True, True, True),
      ('TTF', True, True, False),
      ('TTr', True, True, z3.Bool('r')),
      ('TFT', True, False, True),
      ('TFF', True, False, False),
      ('TFr', True, False, z3.Bool('r')),
      ('TqT', True, z3.Bool('q'), True),
      ('TqF', True, z3.Bool('q'), False),
      ('Tqr', True, z3.Bool('q'), z3.Bool('r')),
      ('FTT', False, True, True),
      ('FTF', False, True, False),
      ('FTr', False, True, z3.Bool('r')),
      ('FFT', False, False, True),
      ('FFF', False, False, False),
      ('FFr', False, False, z3.Bool('r')),
      ('FqT', False, z3.Bool('q'), True),
      ('FqF', False, z3.Bool('q'), False),
      ('Fqr', False, z3.Bool('q'), z3.Bool('r')),
      ('pTT', z3.Bool('p'), True, True),
      ('pTF', z3.Bool('p'), True, False),
      ('pTr', z3.Bool('p'), True, z3.Bool('r')),
      ('pFT', z3.Bool('p'), False, True),
      ('pFF', z3.Bool('p'), False, False),
      ('pFr', z3.Bool('p'), False, z3.Bool('r')),
      ('pqT', z3.Bool('p'), z3.Bool('q'), True),
      ('pqF', z3.Bool('p'), z3.Bool('q'), False),
      ('pqr', z3.Bool('p'), z3.Bool('q'), z3.Bool('r')),
  )
  def test_chains(self, p, q, r):

    def test_fn(a, b, c):
      return (not (a and b and c)) == (not a or (not b) or (not c))

    converted_fn = self.convert(test_fn)
    self.assertTrue(prove(converted_fn(p, q, r)))

  @parameterized.named_parameters(
      ('TTT', True, True, True),
      ('TTF', True, True, False),
      ('TTr', True, True, z3.Bool('r')),
      ('TFT', True, False, True),
      ('TFF', True, False, False),
      ('TFr', True, False, z3.Bool('r')),
      ('TqT', True, z3.Bool('q'), True),
      ('TqF', True, z3.Bool('q'), False),
      ('Tqr', True, z3.Bool('q'), z3.Bool('r')),
      ('FTT', False, True, True),
      ('FTF', False, True, False),
      ('FTr', False, True, z3.Bool('r')),
      ('FFT', False, False, True),
      ('FFF', False, False, False),
      ('FFr', False, False, z3.Bool('r')),
      ('FqT', False, z3.Bool('q'), True),
      ('FqF', False, z3.Bool('q'), False),
      ('Fqr', False, z3.Bool('q'), z3.Bool('r')),
      ('pTT', z3.Bool('p'), True, True),
      ('pTF', z3.Bool('p'), True, False),
      ('pTr', z3.Bool('p'), True, z3.Bool('r')),
      ('pFT', z3.Bool('p'), False, True),
      ('pFF', z3.Bool('p'), False, False),
      ('pFr', z3.Bool('p'), False, z3.Bool('r')),
      ('pqT', z3.Bool('p'), z3.Bool('q'), True),
      ('pqF', z3.Bool('p'), z3.Bool('q'), False),
      ('pqr', z3.Bool('p'), z3.Bool('q'), z3.Bool('r')),
  )
  def test_nesting(self, p, q, r):

    def test_fn(a, b, c):
      return ((a and b) or (b and c)) == ((a or c) and b)

    converted_fn = self.convert(test_fn)
    self.assertTrue(prove(converted_fn(p, q, r)))

  @parameterized.named_parameters(
      ('TTT', True, True, True),
      ('TTF', True, True, False),
      ('TTr', True, True, z3.Bool('r')),
      ('TFT', True, False, True),
      ('TFF', True, False, False),
      ('TFr', True, False, z3.Bool('r')),
      ('TqT', True, z3.Bool('q'), True),
      ('TqF', True, z3.Bool('q'), False),
      ('Tqr', True, z3.Bool('q'), z3.Bool('r')),
      ('FTT', False, True, True),
      ('FTF', False, True, False),
      ('FTr', False, True, z3.Bool('r')),
      ('FFT', False, False, True),
      ('FFF', False, False, False),
      ('FFr', False, False, z3.Bool('r')),
      ('FqT', False, z3.Bool('q'), True),
      ('FqF', False, z3.Bool('q'), False),
      ('Fqr', False, z3.Bool('q'), z3.Bool('r')),
      ('pTT', z3.Bool('p'), True, True),
      ('pTF', z3.Bool('p'), True, False),
      ('pTr', z3.Bool('p'), True, z3.Bool('r')),
      ('pFT', z3.Bool('p'), False, True),
      ('pFF', z3.Bool('p'), False, False),
      ('pFr', z3.Bool('p'), False, z3.Bool('r')),
      ('pqT', z3.Bool('p'), z3.Bool('q'), True),
      ('pqF', z3.Bool('p'), z3.Bool('q'), False),
      ('pqr', z3.Bool('p'), z3.Bool('q'), z3.Bool('r')),
  )
  def test_if(self, p, q, r):

    def test_fn(a, b, c):
      result = None
      if a:
        result = b
      else:
        result = c
      return result

    converted_fn = conversion.convert(test_fn, z3py, [variables, control_flow])
    self.assertTrue(prove(z3.If(p, q, r) == converted_fn(p, q, r)))

  @parameterized.named_parameters(
      ('TTT', True, True, True),
      ('TTF', True, True, False),
      ('TTr', True, True, z3.Bool('r')),
      ('TFT', True, False, True),
      ('TFF', True, False, False),
      ('TFr', True, False, z3.Bool('r')),
      ('TqT', True, z3.Bool('q'), True),
      ('TqF', True, z3.Bool('q'), False),
      ('Tqr', True, z3.Bool('q'), z3.Bool('r')),
      ('FTT', False, True, True),
      ('FTF', False, True, False),
      ('FTr', False, True, z3.Bool('r')),
      ('FFT', False, False, True),
      ('FFF', False, False, False),
      ('FFr', False, False, z3.Bool('r')),
      ('FqT', False, z3.Bool('q'), True),
      ('FqF', False, z3.Bool('q'), False),
      ('Fqr', False, z3.Bool('q'), z3.Bool('r')),
      ('pTT', z3.Bool('p'), True, True),
      ('pTF', z3.Bool('p'), True, False),
      ('pTr', z3.Bool('p'), True, z3.Bool('r')),
      ('pFT', z3.Bool('p'), False, True),
      ('pFF', z3.Bool('p'), False, False),
      ('pFr', z3.Bool('p'), False, z3.Bool('r')),
      ('pqT', z3.Bool('p'), z3.Bool('q'), True),
      ('pqF', z3.Bool('p'), z3.Bool('q'), False),
      ('pqr', z3.Bool('p'), z3.Bool('q'), z3.Bool('r')),
  )
  def test_if_tuple(self, p, q, r):

    def test_fn(a, b, c):
      result = None
      test_result = None
      if a:
        test_result = c
        result = b
      else:
        result = c
        test_result = b
      return result, test_result

    converted_fn = conversion.convert(test_fn, z3py, [variables, control_flow])
    a, b = converted_fn(p, q, r)
    self.assertTrue(prove(z3.If(p, z3.And(q, r), z3.And(q, r)) == z3.And(a, b)))

  def test_eight_queens(self):
    # See https://ericpony.github.io/z3py-tutorial/guide-examples.htm

    def test_fn(queens):
      diagonals = []
      for i in range(8):
        for j in range(i):
          result = None
          if i == j:
            result = True
          else:
            result = queens[i] - queens[j] != i - j and queens[i] - queens[
                j] != j - 1

          diagonals.append(result)

      return diagonals

    queens = [z3.Int('queens_%i' % (i + 1)) for i in range(8)]
    ranks = [z3.And(1 <= queens[i], queens[i] <= 8) for i in range(8)]
    files = [z3.Distinct(queens)]
    converted_fn = conversion.convert(test_fn, z3py,
                                      [logical_ops, variables, control_flow])
    diagonals = converted_fn(queens)
    self.assertTrue(can_solve(ranks + files + diagonals))

  def test_eight_queens_optimized(self):
    def test_fn():
      queens = [z3.Int('queens_%i' % (i + 1)) for i in range(8)]
      ranks = [1 <= queens[i] and queens[i] <= 8 for i in range(8)]
      files = [z3.Distinct(queens)]
      diagonals = []
      for i in range(8):
        for j in range(i):
          if i != j:
            diagonals.append(abs(queens[i] - queens[j]) != abs(i - j))

      return ranks, files, diagonals

    converted_fn = conversion.convert(test_fn, z3py, [logical_ops, functions])
    ranks, files, diagonals = converted_fn()
    self.assertTrue(can_solve(ranks + files + diagonals))

  def test_sudoku(self):

    def get_instance():
      # sudoku instance, we use '0' for empty cells
      return ((0, 0, 0, 0, 9, 4, 0, 3, 0), (0, 0, 0, 5, 1, 0, 0, 0, 7),
              (0, 8, 9, 0, 0, 0, 0, 4, 0), (0, 0, 0, 0, 0, 0, 2, 0, 8),
              (0, 6, 0, 2, 0, 1, 0, 5, 0), (1, 0, 2, 0, 0, 0, 0, 0, 0),
              (0, 7, 0, 0, 0, 0, 5, 2, 0), (9, 0, 0, 0, 6, 5, 0, 0,
                                            0), (0, 4, 0, 9, 7, 0, 0, 0, 0))

    def z3_sudoku():
      # See https://ericpony.github.io/z3py-tutorial/guide-examples.htm
      # 9x9 matrix of integer variables
      x = [[z3.Int('x_%s_%s' % (i + 1, j + 1))
            for j in range(9)]
           for i in range(9)]

      # each cell contains a value in {1, ..., 9}
      cells_c = [
          z3.And(1 <= x[i][j], x[i][j] <= 9) for i in range(9) for j in range(9)
      ]

      # each row contains a digit at most once
      rows_c = [z3.Distinct(x[i]) for i in range(9)]

      # each column contains a digit at most once
      cols_c = [z3.Distinct([x[i][j] for i in range(9)]) for j in range(9)]

      # each 3x3 square contains a digit at most once
      sq_c = [
          z3.Distinct(
              [x[3 * i0 + i][3 * j0 + j]
               for i in range(3)
               for j in range(3)])
          for i0 in range(3)
          for j0 in range(3)
      ]

      sudoku_c = cells_c + rows_c + cols_c + sq_c
      instance = get_instance()
      instance_c = [
          z3.If(instance[i][j] == 0, True, x[i][j] == instance[i][j])
          for i in range(9)
          for j in range(9)
      ]

      return sudoku_c + instance_c

    def naive_sudoku():
      # 9x9 matrix of integer variables
      x = [[z3.Int('x_%s_%s' % (i + 1, j + 1))
            for j in range(9)]
           for i in range(9)]

      # each cell contains a value in {1, ..., 9}
      cells_c = [
          1 <= x[i][j] and x[i][j] <= 9 for i in range(9) for j in range(9)
      ]

      # each row contains a digit at most once
      rows_c = [z3.Distinct(x[i]) for i in range(9)]

      # each column contains a digit at most once
      cols_c = [z3.Distinct([x[i][j] for i in range(9)]) for j in range(9)]

      # each 3x3 square contains a digit at most once
      sq_c = [
          z3.Distinct(
              [x[3 * i0 + i][3 * j0 + j]
               for i in range(3)
               for j in range(3)])
          for i0 in range(3)
          for j0 in range(3)
      ]

      sudoku_c = cells_c + rows_c + cols_c + sq_c
      instance = get_instance()
      instance_c = []

      for i in range(9):
        for j in range(9):
          if instance[i][j] == 0:
            instance_c.append(True)
          else:
            instance_c.append(x[i][j] == instance[i][j])

      return sudoku_c + instance_c

    def optimized_sudoku():
      # 9x9 matrix of integer variables
      x = [[z3.Int('x_%s_%s' % (i + 1, j + 1))
            for j in range(9)]
           for i in range(9)]

      # each cell contains a value in {1, ..., 9}
      cells_c = [
          1 <= x[i][j] and x[i][j] <= 9 for i in range(9) for j in range(9)
      ]

      # each row contains a digit at most once
      rows_c = [z3.Distinct(x[i]) for i in range(9)]

      # each column contains a digit at most once
      cols_c = [z3.Distinct([x[i][j] for i in range(9)]) for j in range(9)]

      # each 3x3 square contains a digit at most once
      sq_c = [
          z3.Distinct(
              [x[3 * i0 + i][3 * j0 + j]
               for i in range(3)
               for j in range(3)])
          for i0 in range(3)
          for j0 in range(3)
      ]

      sudoku_c = cells_c + rows_c + cols_c + sq_c
      instance = get_instance()
      instance_c = []

      for i in range(9):
        for j in range(9):
          if instance[i][j] != 0:
            instance_c.append(x[i][j] == instance[i][j])

      return sudoku_c + instance_c

    converted_naive = conversion.convert(naive_sudoku, z3py,
                                         [logical_ops, functions])
    converted_opt = conversion.convert(optimized_sudoku, z3py,
                                       [logical_ops, functions])

    self.assertEqual(can_solve(converted_naive()), can_solve(z3_sudoku()))
    self.assertEqual(can_solve(converted_opt()), can_solve(z3_sudoku()))


if __name__ == '__main__':
  test.main()
