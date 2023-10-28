nohup python -u train_test.py --device cuda:0 --alpha 0.5 --dataset NCI1 --loss_log 0 > logs/NCI1_out5.log  2>&1 &
nohup python -u train_test.py --device cuda:1 --alpha 0.4 --dataset NCI1 --loss_log 1 > logs/NCI1_out4.log  2>&1 &
nohup python -u train_test.py --device cuda:2 --alpha 0.6 --dataset NCI1 --loss_log 2 > logs/NCI1_out6.log  2>&1 &
nohup python -u train_test.py --device cuda:3 --alpha 0.8 --dataset NCI1 --loss_log 3 > logs/NCI1_out8.log  2>&1 &
