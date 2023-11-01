Dataset="PROTEINS"


nohup python -u train_test.py --device cuda:0 --alpha 0.5 --dataset ${Dataset} --loss_log 0 > logs/${Dataset}_out5.log  2>&1 &
nohup python -u train_test.py --device cuda:1 --alpha 0.4 --dataset ${Dataset} --loss_log 1 > logs/${Dataset}_out4.log  2>&1 &
nohup python -u train_test.py --device cuda:2 --alpha 0.6 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out6.log  2>&1 &
nohup python -u train_test.py --device cuda:3 --alpha 0.8 --dataset ${Dataset} --loss_log 3 > logs/${Dataset}_out8.log  2>&1 &
