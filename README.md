# FSAKE: Few-shot graph learning via adaptive neighbor class knowledge embedding
Implementation of FSAKE on Python3.8, Pytorch 1.10.0
## miniImagenet, 5way 5shot 2pooling 5gcn
'''
python eval.py --device cuda:1 --dataset mini --num_ways 5 --num_shots 5 --transductive True --pool_mode kn --unet_mode addold
'''

## Acknowledgment

Our project references the codes in the following repos.
- [HGNN](https://github.com/smartprobe/HGNN)
- [few-shot-gnn](https://github.com/vgsatorras/few-shot-gnn)


