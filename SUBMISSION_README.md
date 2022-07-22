# My submission Readme

We combine several pre-processing steps and incorporate them into the model so that the entire pipeline can run on the GPU instead of having a seperate pre-processing pipeline.
The initial model layers use a median filter and PCEN to normalize the volume levels of foreground and background noises so the model receives a uniform input.  
During training the model is exposed to a variety of random noise sources and both time and frequency masking are used for regularization.
The model backbone is a ConvNeXt CNN.


Reverted to checkpoint used in MIDS Capstone April '22

```python

```
