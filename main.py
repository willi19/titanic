from nn import NN
from dataloader import PassengerDataSet
import torch
import pandas as pd

batch_size = 10

train_data = PassengerDataSet("train.csv", True, False)
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)

valid_data = PassengerDataSet("train.csv", True, True)

test_data = PassengerDataSet("test.csv", False)

model = NN(batch_size, 8)

loss_func = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


for epoch in range(1000):
    train_loss = 0
    for features, labels in train_loader:
        output = model(features)
        loss = loss_func(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    if epoch%100 == 0:
        tot = len(valid_data)
        accuracy = 0
        predict = model.predict(valid_data.data)
        for i in range(tot):
            if predict[i] == valid_data.target[i]:
                accuracy += 1
        print(str(train_loss)+" "+str(epoch))
        print("accuracy: "+str(accuracy/tot))        

predict = model.predict(test_data.data)
df_out = pd.DataFrame(columns = ["PassengerId","Survived"])
for i in range(892, 1310):
    df_out.loc[i] = {"PassengerId": i, "Survived": predict[i-892]}

df_out.to_csv("answer.csv", mode = 'w', index = False)