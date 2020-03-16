# Visual-Dialog
## Visual Dialog Agents for image retrieval with Memory Network (work for PR)
Xinyu Lyu
### Our contribution includes:
1. Retrieve a specific picture from the picture dataset by talking between two robots.
2. Use the HCIAE Memory Network and use the attention mechanism to fuse picture visual information and dialogue text information to improve the quality of the dialogue.
3. Design the reward and punishment mechanism with reinforcement learning, to make the dialogue of the dialogue robot towards the target and carry out dialogue tasks and retrieve the required pictures.
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
   Q-Bot
   A-Bot
   Image Retrieval
