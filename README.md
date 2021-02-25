# Visual-Dialog
## GuessWhich? Visual dialog with attentive memory network (Pattern Recognition)
LeiZhao, XinyuLyu, JingkuanSong, LianliGao
*Paper: https://www.sciencedirect.com/science/article/pii/S0031320321000108?utm_campaign=STMJ_AUTH_SERV_PUBLISHED&utm_medium=email&utm_acid=128326383&SIS_ID=&dgcid=STMJ_AUTH_SERV_PUBLISHED&CMX_ID=&utm_in=DM121054&utm_source=AC_*
### Our contribution includes:
1. Retrieve a specific picture from the picture dataset by talking between two robots.
2. Use the HCIAE Memory Network and use the attention mechanism to fuse picture visual information and dialogue text information to improve the quality of the dialogue.
3. Design the reward and punishment mechanism with reinforcement learning, to make the dialogue of the dialogue robot towards the target and carry out dialogue tasks and retrieve the required pictures.
### Framework:
   1. Qbot![](https://github.com/XinyuLyu/Visual-Dialog/blob/master/test_results/1-s2.0-S0031320321000108-gr1_lrg.jpg)
### Dataset: Visdial V0.9   

### Environments:
  1. Python 3.6
  2. Pytorch 0.3.1

### Hardware:
  1. Our model was trained on a GTX 1080 GPU with 32GB RAM.
  
### Codes: 
  1. models : Q-bot & A-bot
  2. dataloader
  3. utils
  4. train.py
  5. evaluate.py

### Test result:   
   1. Qbot![](https://github.com/XinyuLyu/Visual-Dialog/blob/master/test_results/Xnip2020-03-16_14-44-51.jpg)
   2. Abot![](https://github.com/XinyuLyu/Visual-Dialog/blob/master/test_results/Xnip2020-03-16_14-18-24.jpg)
   
   3. Visualization
    ![](https://github.com/XinyuLyu/Visual-Dialog/blob/master/test_results/1-s2.0-S0031320321000108-gr2_lrg.jpg)
    ![](https://github.com/XinyuLyu/Visual-Dialog/blob/master/test_results/1-s2.0-S0031320321000108-gr4_lrg.jpg)

 
