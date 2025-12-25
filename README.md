# MDETR-G (Modulated Detection Transformer — Geospatial)

MDETR-G is a geospatially adapted variant of **MDETR** for **text-conditioned object detection** in overhead / geospatial imagery. 
It keeps MDETR’s end-to-end “detect what the text describes” formulation, but updates the vision + attention stack to better handle high-resolution 
geospatial scenes and small targets.


## What’s different vs. baseline MDETR?

MDETR-G modifies the original MDETR design in a few key ways:

- **Deformable attention** for more efficient multi-scale spatial reasoning (especially helpful in high-resolution imagery).
- **Swin Transformer backbone** tuned for aerial/overhead imagery (stronger hierarchical feature extraction than natural-image backbones). 
- **Learnable contrastive temperature (τ)** in the text–image alignment objective (rather than a fixed constant).  
- **Shallower transformer (3 encoder / 3 decoder layers)** to improve training/inference efficiency. :contentReference[oaicite:5]{index=5}  

Under the hood, MDETR-G still uses:
- Hungarian matching (set prediction),
- box regression losses (L1 + GIoU),
- soft token prediction,
- contrastive alignment between token and region embeddings. :contentReference[oaicite:6]{index=6}  


## Core capability

Given an image and a natural-language query, MDETR-G predicts bounding boxes corresponding to the referenced objects.


## Evaluation note (terminology)

