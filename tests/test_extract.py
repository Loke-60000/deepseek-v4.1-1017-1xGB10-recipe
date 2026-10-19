import unittest

try:
    import torch

    from recipe.ablate.extract import DirectionExtractor, Fp8Writer
except ImportError:
    torch = None


class FakeExtractor(DirectionExtractor):
    def __init__(self, writers, rank):
        self.writers = writers
        self.layers = list(writers)
        self.rank = rank

    def writer(self, layer):
        return self.writers[layer]


@unittest.skipUnless(torch, "torch")
class ExtractTest(unittest.TestCase):
    def test_recovers_directions_and_strengths(self):
        torch.manual_seed(0)
        d, n = 64, 96
        r = torch.linalg.qr(torch.randn(d, 2))[0]
        lams = {5: torch.tensor([1.2, 0.4]), 6: torch.tensor([0.9, 0.7])}
        writers = {}
        for layer, lam in lams.items():
            w = torch.randn(d, n)
            edited = w - r @ torch.diag(lam) @ r.T @ w
            writers[layer] = (w, edited - w)
        found = FakeExtractor(writers, rank=2).run()
        q = found.directions.T
        for layer, lam in lams.items():
            want = r @ torch.diag(lam) @ r.T
            got = q @ found.mix_attn[layer] @ q.T
            self.assertTrue(torch.allclose(got, want, atol=1e-4))
        self.assertTrue(all(c > 0.9999 for c in found.captured))

    def test_fp8_block_scales(self):
        weight = torch.full((32, 64), 1.5).to(torch.float8_e4m3fn)
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)
        out = Fp8Writer.dequantize(weight.view(torch.uint8).numpy().tobytes(), scale.numpy().tobytes(), (32, 64))
        self.assertEqual(out[0, 0].item(), 1.5)
        self.assertEqual(out[0, 40].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
