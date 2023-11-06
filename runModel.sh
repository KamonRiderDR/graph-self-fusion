###
 # @Description: 
 # @Author: Rui Dong
 # @Date: 2023-11-03 11:29:53
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2023-11-06 18:47:43
### 

Dataset="IMDB-MULTI"
alpha=0.4
ALPHA=${alpha}


nohup python -u main.py\
    --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 6\
    --batch_size 128 --epoches 800  --device cuda:0 --alpha ${alpha} --dataset ${Dataset}\
    --loss_log 2 > logs/${Dataset}_out${ALPHA}_2.log  2>&1 &

# nohup python -u main.py\
#     --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 6\
#     --batch_size 128 --epoches 800  --device cuda:0 --alpha ${alpha} --dataset ${Dataset}\
#     --loss_log 2 > logs/${Dataset}_out${ALPHA}_2.log  2>&1 &

# nohup python -u main.py --epoches 1000  --device cuda:2 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_3.log  2>&1 &
# nohup python -u main.py --epoches 1000  --device cuda:2 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_4.log  2>&1 &
