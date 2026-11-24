import unittest

from mizan.estimator import WeighInput, lora_param_count, param_counts, verdict, weigh
from mizan.models import PRESETS


class ParamCountTests(unittest.TestCase):
    def test_llama_8b_close_to_published_size(self):
        n = param_counts(PRESETS["llama-3.1-8b"])["total"]
        self.assertTrue(7.8e9 < n < 8.3e9, n)

    def test_qwen_7b_close_to_published_size(self):
        n = param_counts(PRESETS["qwen2.5-7b"])["total"]
        self.assertTrue(7.3e9 < n < 7.9e9, n)

    def test_tied_embeddings_remove_lm_head(self):
        spec = PRESETS["llama-3.2-1b"]
        counts = param_counts(spec)
        self.assertEqual(counts["lm_head"], 0)
        self.assertEqual(counts["total"], counts["transformer"] + counts["embedding"] + spec.hidden_size)


class LoraTests(unittest.TestCase):
    def test_rank_16_all_targets_is_tens_of_millions(self):
        n = lora_param_count(PRESETS["llama-3.1-8b"], 16)
        self.assertTrue(35e6 < n < 50e6, n)

    def test_rank_scales_linearly(self):
        spec = PRESETS["llama-3.1-8b"]
        self.assertEqual(lora_param_count(spec, 32), 2 * lora_param_count(spec, 16))

    def test_unknown_target_raises(self):
        with self.assertRaises(ValueError):
            lora_param_count(PRESETS["llama-3.1-8b"], 8, ("q", "nope"))


class WeighTests(unittest.TestCase):
    def cfg(self, **kwargs):
        base = dict(spec=PRESETS["llama-3.1-8b"], seq_len=512, batch_size=1)
        base.update(kwargs)
        return WeighInput(**base)

    def test_base_4bit_weight_dominates(self):
        r = weigh(self.cfg())
        self.assertGreater(r.component("base_weights") / r.total_mib, 0.5)

    def test_8gb_card_fits_8b_qlora_at_512_tokens(self):
        r = weigh(self.cfg())
        status, _ = verdict(r, 8)
        self.assertEqual(status, "OK")

    def test_4gb_card_rejected(self):
        r = weigh(self.cfg())
        status, _ = verdict(r, 4)
        self.assertEqual(status, "OVER")

    def test_checkpointing_reduces_activations(self):
        plain = weigh(self.cfg(gradient_checkpointing=False))
        ckpt = weigh(self.cfg(gradient_checkpointing=True))
        self.assertLess(ckpt.component("activations"), plain.component("activations"))

    def test_longer_sequence_costs_more(self):
        short = weigh(self.cfg(seq_len=512))
        long = weigh(self.cfg(seq_len=4096))
        self.assertGreater(long.total_mib, short.total_mib)

    def test_flash_attention_saves_memory_at_long_context(self):
        flash = weigh(self.cfg(seq_len=2048, flash_attention=True))
        eager = weigh(self.cfg(seq_len=2048, flash_attention=False))
        self.assertLess(flash.component("activations"), eager.component("activations"))

    def test_full_finetune_costs_far_more_than_lora(self):
        lora = weigh(self.cfg())
        full = weigh(self.cfg(full_finetune=True))
        self.assertGreater(full.total_mib, lora.total_mib * 2)

    def test_quantization_lowers_base_weights(self):
        q4 = weigh(self.cfg(quant="4bit"))
        bf16 = weigh(self.cfg(quant="bf16"))
        self.assertLess(q4.component("base_weights"), bf16.component("base_weights"))

    def test_zero_batch_rejected(self):
        with self.assertRaises(ValueError):
            weigh(self.cfg(batch_size=0))

    def test_unknown_quant_rejected(self):
        with self.assertRaises(ValueError):
            weigh(self.cfg(quant="3bit"))

    def test_json_shape(self):
        d = weigh(self.cfg()).as_dict()
        self.assertIn("total_gib", d)
        self.assertIn("components_gib", d)
        self.assertIn("parameters", d)


if __name__ == "__main__":
    unittest.main()
