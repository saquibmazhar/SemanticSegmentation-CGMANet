# SemanticSegmentation-CGMANet
This is the implementation of our work "Context Guided Multiscale Attention for Realtime Semantic Segmentation". 
The repository has three training scripts: 
* `train_enc.py` to train only the encoder
* `train_dec.py` to train the full encoder-decoder network
* `train_fine_tune.py` to fine-tune the fully trained model. <br/>

<br/>
Download the Cityscapes and CamVid datasets and save them in their corresponding folders in [dataset](/dataset). 

```python
import torch
```
