## Dependencies
Please refer to https://github.com/Shen-Lab/CPAC/tree/main/cross_modality_torch#dependencies for environment.

## Pre-training Strategies for Cross-Modality Models:
```
mkdir ./weights
```

Mask language modeling pre-training for 1D encoder:
```
python main_pretrain.py --ss_seq mask --ss_graph none --p_seq_mask ${MASK_RETIO}
```
where ```${MASK_RETIO}``` is selected from {0.05, 0.15, 0.25}.

Graph completion pre-training for 2D encoder:
```
python main_pretrain.py --ss_seq none --ss_graph mask --p_graph_mask ${MASK_RETIO}
```
where ```${MASK_RETIO}``` is selected from {0.05, 0.15, 0.25}.

Joint pre-training for cross-modality models:
```
python main_pretrain.py --ss_seq mask --ss_graph none --p_seq_mask ${MASK_RETIO} --p_graph_mask ${MASK_RETIO}
```

## Acknowledgements

The graph completion implementation is reference to https://github.com/Shen-Lab/SS-GCNs.

colab

.15 .15 seqsub0.5 drop0

epoch0 5.72

epoch1 5.67

2 5.67

3 5.66

seqsub default drop .1

4 5.66

5 5.65



!python cuda.py --batch_size=85 --resume=0 --epoch=10 --ss_seq none --ss_graph mask --p_seq_mask 0.15 --p_graph_mask 0.15 --p_graph_sub=0.1 --resume=0 --p_graph_drop=0.1

!python cuda.py --batch_size=85 --resume=0 --epoch=10 --ss_seq mask --ss_graph none --p_seq_mask 0.15 --p_graph_mask 0.15 --p_graph_sub=0.1 --resume=0 --p_graph_drop=0.1

2.79->2.77
