from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'axes.spines.top':False, 'axes.spines.right':False,
                     'figure.dpi':150, 'axes.titleweight':'bold'})
(ROOT/'figures').mkdir(exist_ok=True)

import torch
from models import scaled_dot_product_attention
torch.manual_seed(42)
q,k,v=[torch.randn(1,8,16) for _ in range(3)]
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,causal,title in zip(axes,[False,True],['Full attention','Causal attention']):
    _,weights=scaled_dot_product_attention(q,k,v,causal_mask=causal)
    ax.imshow(weights[0].detach().numpy(),cmap='Blues',vmin=0,vmax=1)
    ax.set(xlabel='Key position',ylabel='Query position',title=title)
fig.savefig(ROOT/'figures/attention.svg')
print('Generated attention.svg from seeded synthetic tensors.')
