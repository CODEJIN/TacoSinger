export CUDA_HOME=/usr/local/cuda
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=32 python -m torch.distributed.launch --nproc_per_node=4 Train.py --hyper_parameters Hyper_Parameters.yaml --port 54321