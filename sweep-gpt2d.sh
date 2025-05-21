#Fig 4

# Single node 8 GPUs, sequential sweep
# sh sweep-gpt2d.sh | python sweep.py sweeps/gpt2d 
# nohup make -f sweeps/gpt2d/Makefile -j 1 -k

cmd='torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --wandb_run_name=gpt2d'

echo $cmd --qat=b8p4 --qat_rnd=tne
echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=sr
echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=srf
echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=srff
echo $cmd
echo $cmd --qat=float16 --qat_rnd=tne

echo $cmd --qat=b8p4 --qat_srn=2 --qat_rnd=sr
echo $cmd --qat=b8p4 --qat_srn=2 --qat_rnd=srf
echo $cmd --qat=b8p4 --qat_srn=2 --qat_rnd=srff

echo $cmd --qat=b8p4 --qat_srn=4 --qat_rnd=sr
echo $cmd --qat=b8p4 --qat_srn=4 --qat_rnd=srf
echo $cmd --qat=b8p4 --qat_srn=4 --qat_rnd=srff

# cmd='torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py'
# echo $cmd --qat=float16 --qat_rnd=tne
# echo $cmd --qat=b8p4 --qat_rnd=tne
# echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=sr --qat_scale=0.0 --qat_start_iter=
# echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=srf
# echo $cmd --qat=b8p4 --qat_srn=3 --qat_rnd=srff
