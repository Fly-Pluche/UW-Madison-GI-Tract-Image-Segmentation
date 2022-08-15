export CUDA_VISIBLE_DEVICES=0;
nohup python main.py > logs/train_efficientnet-b1_fold4.log 2>&1 &
