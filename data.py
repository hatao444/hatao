
from torch.utils.data import Dataset

class PeopleDaily(Dataset):
    def __init__(self, data_file):
        self.data,self.categories= self.load_data(data_file)




    def load_data(self, data_file):#返回一个字典，一个数据包含句子和标签
        Data={}
        categories= set()
        with open(data_file, 'rt',encoding='utf-8') as f:
            for inx,line in enumerate(f.read().split('\n\n')):
                if not line:
                    break
                sentence,labels='',[]
                for i,item in enumerate(line.split('\n')):
                    char,tag = item.split(' ')
                    sentence+=char
                    if tag.startswith('B'):
                        labels.append([i,i,char,tag[2:]])
                        categories.add(tag[2:])
                    elif tag.startswith('I') and labels:
                        labels[-1][2]+=char
                        labels[-1][1]=i

                Data[inx]={
                    'sentence':sentence,
                    'labels':labels
                }
        return Data,categories
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_data = PeopleDaily('./example.train.txt')
id2label = {0:'O'}
for c in list(sorted(train_data.categories)):
    id2label[len(id2label)] = f"B-{c}"
    id2label[len(id2label)] = f"I-{c}"
label2id = {v: k for k, v in id2label.items()}





