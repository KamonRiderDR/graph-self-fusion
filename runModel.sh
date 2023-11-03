Dataset="NCI1"
###
 # @Description: 
 # @Author: Rui Dong
 # @Date: 2023-11-03 11:29:53
 # @LastEditors: Rui Dong
 # @LastEditTime: 2023-11-03 14:24:30
### 


nohup python -u main.py --epoches 1000  --device cuda:0 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_1.log  2>&1 &
nohup python -u main.py --epoches 1000  --device cuda:1 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_2.log  2>&1 &
nohup python -u main.py --epoches 1000  --device cuda:2 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_3.log  2>&1 &
nohup python -u main.py --epoches 1000  --device cuda:3 --alpha 0.8 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out8_1.log  2>&1 &
