# krplatz's GPT & MLP implementation

Howdy, here's my own implementation for MLP and GPT. It's a sloppy mess right now, but I'm proud of the work I made considering minimal LM support and actually doing the research and docs reading for this. Special mention to Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master), this would've been impossible without it.

## Quick Stats
MLP
* Parameters: 101,770
* Final Loss: **0.041**
* Final Validation Accuracy: **0.977**

GPT
* Parameters: 10,746,689
* Final Loss: **2.672**
* Final Validation Accuracy: **0.515**
* Perplexity (fp32): **19.248**
* Perplexity (int4): **18.991**