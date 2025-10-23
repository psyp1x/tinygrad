import unittest
import numpy as np
import torch
from tinygrad import Tensor

# Targeted tests for the historical alternative Tensor._pool path.
# That path used to be taken when dilation==1 and all(k <= s), with an internal pad/reshape flow.
# We verify a few edge-y shapes where padding was required to complete the last window, plus ceil_mode.

class TestPoolUnifiedPath(unittest.TestCase):
  def _cmp(self, tx, ty, rtol=1e-5, atol=1e-6):
    a, b = tx.detach().cpu().numpy(), ty.numpy()
    self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol), f"mismatch\nTorch:\n{a}\nTiny:\n{b}")

  def test_max_pool2d_pad_required_k2_s3(self):
    # shape chosen so o*s - i > 0 along each spatial axis; e.g., H=W=5, k=2, s=3 => o=2, o*s=6 > i=5
    x = torch.randn(1,1,5,5)
    k, s = (2,2), (3,3)
    tx = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=s)
    ty = Tensor.max_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s)
    self._cmp(tx, ty)

  def test_avg_pool2d_pad_required_k2_s3_include_pad(self):
    x = torch.randn(2,3,5,5)
    k, s = (2,2), (3,3)
    tx = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=s, count_include_pad=True)
    ty = Tensor.avg_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s, count_include_pad=True)
    self._cmp(tx, ty, rtol=1e-5)

  def test_avg_pool2d_pad_required_k2_s3_exclude_pad(self):
    x = torch.randn(2,3,5,5)
    k, s = (2,2), (3,3)
    tx = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=s, count_include_pad=False)
    ty = Tensor.avg_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s, count_include_pad=False)
    self._cmp(tx, ty, rtol=1e-5)

  def test_max_pool2d_k_eq_s(self):
    x = torch.randn(1,2,9,12)
    k, s = (3,3), (3,3)
    tx = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=s)
    ty = Tensor.max_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s)
    self._cmp(tx, ty)

  def test_avg_pool2d_k_eq_s(self):
    x = torch.randn(1,2,9,12)
    k, s = (3,3), (3,3)
    tx = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=s)
    ty = Tensor.avg_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s)
    self._cmp(tx, ty, rtol=1e-5)

  def test_max_pool2d_pad_required_ceil_mode(self):
    # ceil_mode changes output size logic; also triggers edge window behavior
    x = torch.randn(1,1,5,5)
    k, s = (2,2), (3,3)
    tx = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=s, ceil_mode=True)
    ty = Tensor.max_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s, ceil_mode=True)
    self._cmp(tx, ty)

  def test_avg_pool2d_pad_required_ceil_mode_include_pad(self):
    x = torch.randn(1,1,5,5)
    k, s = (2,2), (3,3)
    tx = torch.nn.functional.avg_pool2d(x, kernel_size=k, stride=s, ceil_mode=True, count_include_pad=True)
    ty = Tensor.avg_pool2d(Tensor(x.numpy()), kernel_size=k, stride=s, ceil_mode=True, count_include_pad=True)
    self._cmp(tx, ty, rtol=1e-5)

if __name__ == "__main__":
  unittest.main()
