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
"""Benchmark comparing eight queens implementations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyctr.api import conversion
from pyctr.examples.models import eight_queens
from pyctr.examples.sysml2019 import benchmark_base
from pyctr.examples.z3py import z3py
from pyctr.transformers.virtualization import control_flow
from pyctr.transformers.virtualization import functions
from pyctr.transformers.virtualization import logical_ops
import z3


def can_solve(f):
  s = z3.Solver()
  s.add(f)
  return s.check() == z3.sat


class QueensBenchmark(benchmark_base.ReportingBenchmark):
  """Runs benchmarks for variants of eight queens."""

  def _benchmark_z3_queens(self):
    constraints = eight_queens.z3_queens()
    self.time_execution('queens_z3', lambda: can_solve(constraints))

  def _benchmark_naive_queens(self):
    converted_fn = conversion.convert(eight_queens.naive, z3py,
                                      [logical_ops, control_flow])
    constraints = converted_fn()
    self.time_execution('queens_naive', lambda: can_solve(constraints))

  def _benchmark_simplified_queens(self):
    converted_fn = conversion.convert(eight_queens.simplified, z3py,
                                      [logical_ops, functions])
    constraints = converted_fn()
    self.time_execution('queens_simplified', lambda: can_solve(constraints))

  def benchmark_queens(self):
    self._benchmark_z3_queens()
    self._benchmark_naive_queens()
    self._benchmark_simplified_queens()


if __name__ == '__main__':
  QueensBenchmark().benchmark_queens()
