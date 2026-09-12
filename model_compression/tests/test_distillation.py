import unittest


class DistillationTests(unittest.TestCase):
    def test_loss_matches_weighted_ce_and_kl(self):
        import torch
        import torch.nn.functional as F

        from model_compression.modules.distillation import distillation_loss

        student = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        teacher = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        labels = torch.tensor([1, 0])
        temperature, alpha = 2.0, 0.25
        expected = (1 - alpha) * F.cross_entropy(student, labels) + alpha * temperature**2 * F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        )
        actual = distillation_loss(student, teacher, labels, temperature=temperature, alpha=alpha)
        self.assertTrue(torch.allclose(actual, expected))

    def test_invalid_temperature_is_rejected(self):
        from model_compression.modules.distillation import validate_distillation_config

        with self.assertRaises(ValueError):
            validate_distillation_config({"temperature": 0})


if __name__ == "__main__":
    unittest.main()
