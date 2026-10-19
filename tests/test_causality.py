import unittest
import torch
from models import scaled_dot_product_attention,DecoderOnlyTransformer
class CausalTests(unittest.TestCase):
    def test_attention_normalization_and_mask(self):
        torch.manual_seed(2)
        q,k,v=[torch.randn(2,6,16) for _ in range(3)]
        _,a=scaled_dot_product_attention(q,k,v,True)
        self.assertTrue(torch.allclose(a.sum(-1),torch.ones(2,6)))
        self.assertEqual(torch.triu(a,diagonal=1).abs().max().item(),0)
    def test_future_inputs_cannot_change_prefix(self):
        torch.manual_seed(7)
        model=DecoderOnlyTransformer(16,4,32,2,37,dropout=0).eval()
        x=torch.randn(8,2,16)
        changed=x.clone(); changed[4:]+=torch.randn_like(changed[4:])*20
        with torch.no_grad(): a,b=model(x),model(changed)
        self.assertEqual(tuple(a.shape),(8,2,37))
        self.assertTrue(torch.allclose(a[:4],b[:4],atol=1e-6))
        self.assertFalse(torch.allclose(a[4:],b[4:]))
if __name__=='__main__': unittest.main()
