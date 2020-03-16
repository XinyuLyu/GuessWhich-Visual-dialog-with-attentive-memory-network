import json
#/hhd/lvxinyu/aqm_plus/dialog_output/results/results1.json
#/hhd/lvxinyu/aqm_plus/dialog_output/results/results_all.json

#/hhd/lvxinyu/visdial-pytorch/dialog_output/results/results0.json
#/hhd/lvxinyu/visdial-pytorch/dialog_output/results/results.json
#/hhd/lvxinyu/visdial-pytorch/dialog_output/results/results0.json
import json
with open("/home/lvxinyu/results.json","r") as load_f: #dtype 换成train或val或test
     load_dict = json.load(load_f)
     list = []
     for i in load_dict['data']:
          list.append(i['image_id'])
     print()


with open("/hhd/lvxinyu/visdial-pytorch/dialog_output/results/results.json","r") as load_f:
     load_dict = json.load(load_f)
     print()

