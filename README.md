# Unsupervised learning of text line segmentation by differentiating coarse patterns 

We present an unsupervised deep learning method that embeds document image patches to a compact 
Euclidean space where distances correspond to a coarse text line pattern similarity. Once this 
space has been produced text line segmentation can be easily implemented using standard techniques 
with the embedded feature vectors. To train, we extract random pairs of document image patches with
the assumption that neighbour patches contain similar coarse trend oftext lines whereas if one of 
them is rotated they contain different coarse trend of text lines. Doing well on this task requires 
the model to learn to recognize the text lines and their salient parts. The benefit of 
our approach is zero manual labelling effort. 
