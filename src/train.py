from WikiHow_Dataset import WikiHow_Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
        
def train(model, model_name, train_set, val_set, batch_size, num_epochs, device, criterion, path, resume_training = False, save_epoch = 5):

    trainloader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    validloader = DataLoader(val_set, batch_size=batch_size,shuffle=True)

    # Maybe changed - Due to use different optimizers for encoder and decoder, e.g. BERTSUMABS with 2 different Adam
    params = model.parameters()
    optimizer = optim.Adam(params, lr=0.001)

    train_loss = []
    valid_loss = []
    prev_epoch = 0

    if resume_training:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']
        train_loss = checkpoint['training_loss']
        valid_loss = checkpoint['validation_loss']

    model.to(device)

    min_valid_loss = float('inf')

    counter = 0
    for epoch in range(prev_epoch, num_epochs):
        
        training_loss = 0

        for i, train_data in enumerate(trainloader):

            inputs, labels = train_data
            inputs, labels = inputs.to(device), labels.to(device) 

            optimizer.zero_grad()

            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'epoch {epoch+1}, training loss = {training_loss/(i+1)}')
        train_loss.append(running_loss/(i+1))

        running_loss = 0
        for i, val_data in enumerate(validloader):

            inputs, labels = val_data

            inputs, labels = inputs.to(device), labels.to(device)  

            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            running_loss += loss.item()  

        # Save the best model
        if running_loss/(i+1) < min_valid_loss:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = True, save model')  
            torch.save(model.state_dict(), f'{model_name}.pt')    
            min_valid_loss = running_loss/(i+1)
        else:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = False, do not save model')
        
        counter += 1
        
        # Regularly save models between save_epoch epochs, for resuming training
        if counter == save_epoch:
            counter = 0
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': valid_loss,
            }, path)
    
    return model, train_loss, valid_loss
            

