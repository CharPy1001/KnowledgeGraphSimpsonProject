import torch
from torch.utils.data import DataLoader
from BERT import MyDataset, BertRelationExtract


device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_dataset = MyDataset(data_type='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = BertRelationExtract().to(device)
state_dict = torch.load('best_pts/best.pt')
model.load_state_dict(state_dict)

# 验证过程
model.eval()
cls_acc = 0.0

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        model.zero_grad()
        cls = model(data['ids'].to(device), data['att_mask'].to(device), data['h'], data['t'])

        if cls.squeeze().argmax() == data['rel_cls'].squeeze().argmax():
            cls_acc += 1

avr_acc = cls_acc / len(test_dataset)
print(f"Class Accuracy: {avr_acc:.9f}")

