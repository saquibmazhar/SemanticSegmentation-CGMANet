# SemanticSegmentation-CGMANet
This is the implementation of our work "Context Guided Multiscale Attention for Realtime Semantic Segmentation". It proposes a lightweight model for the segmentation of urban road scenes in resource-constrained devices.
The repository has three training scripts: 
* `train_enc.py` to train only the encoder
* `train_dec.py` to train the full encoder-decoder network  
* `train_fine_tune.py` to fine-tune the fully trained model.
<br/>

Download the Cityscapes and CamVid datasets and save them in their corresponding folders in [dataset](/dataset). 

# References
We are grateful to the authors of the following works for sharing the codes: 
* https://github.com/Reagan1311/DABNet
* https://github.com/Junjun2016/APCNet
