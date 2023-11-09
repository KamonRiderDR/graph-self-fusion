###
 # @Description: 
 # @Author: Rui Dong
 # @Date: 2023-11-03 11:29:53
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2023-11-09 18:12:08
### 

Dataset="PTC_FR"
alpha=0.6
ALPHA=${alpha}


nohup python -u main.py\
    --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 6 --ffn_dim 128\
    --lr 0.00001 --weight_decay 0.00005  \
    --batch_size 128  --device cuda:0 --alpha ${alpha} --dataset ${Dataset}\
    --loss_log 2 > logs/${Dataset}_out${ALPHA}_1.log  2>&1 &

# nohup python -u main.py\
#     --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 6 --ffn_dim 128\
#     --lr 0.00001 --weight_decay 0.00005  \
#     --batch_size 128 --epoches 800  --device cuda:0 --alpha ${alpha} --dataset ${Dataset}\
#     --loss_log 2 > logs/${Dataset}_out${ALPHA}_2.log  2>&1 &


# nohup python -u main.py\
#     --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 4 --ffn_dim 128\
#     --batch_size 64 --epoches 800  --device cuda:1 --alpha ${alpha} --dataset ${Dataset}\
#     --loss_log 2 > logs/${Dataset}_out${ALPHA}_2.log  2>&1 &


# nohup python -u main.py\
#     --gcn_hidden 128 --hidden_dim 128 --num_fusion_layers 6\
#     --batch_size 128 --epoches 800  --device cuda:0 --alpha ${alpha} --dataset ${Dataset}\
#     --loss_log 2 > logs/${Dataset}_out${ALPHA}_2.log  2>&1 &

# nohup python -u main.py --epoches 1000  --device cuda:2 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_3.log  2>&1 &
# nohup python -u main.py --epoches 1000  --device cuda:2 --alpha 0.5 --dataset ${Dataset} --loss_log 2 > logs/${Dataset}_out5_4.log  2>&1 &
