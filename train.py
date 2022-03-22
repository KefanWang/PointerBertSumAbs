from torch.utils.data import DataLoader
import torch.optim as optim
import torch

def train_PBSA(
    tokenizer, 
    model, 
    model_name, 
    learning_rate_encoder,
    learning_rate_decoder,
    train_set, 
    val_set, 
    batch_size, 
    num_epochs, 
    device, 
    criterion, 
    checkpoint_path, 
    resume_training, 
    save_epoch,
    last_backup
):

    trainloader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    validloader = DataLoader(val_set, batch_size=batch_size,shuffle=True)

    encoder_params = model.encoder.parameters()
    encoder_optimizer = optim.Adam(encoder_params, lr=learning_rate_encoder)

    decoder_params = model.decoder.parameters()
    decoder_optimizer = optim.Adam(decoder_params, lr=learning_rate_decoder)

    train_loss = []
    valid_loss = []
    prev_epoch = 0
    min_valid_loss = float('inf')

    if resume_training:

        try:
            checkpoint = torch.load(f'{checkpoint_path}_{last_backup}',map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
            prev_epoch = checkpoint['epoch']
            train_loss = checkpoint['training_loss']
            valid_loss = checkpoint['validation_loss']
            min_valid_loss = checkpoint['min_valid_loss'] 
        except:
            checkpoint = torch.load(f'{checkpoint_path}_{1-last_backup}',map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
            prev_epoch = checkpoint['epoch']
            train_loss = checkpoint['training_loss']
            valid_loss = checkpoint['validation_loss']
            min_valid_loss = checkpoint['min_valid_loss'] 
            
        del checkpoint

    model.to(device)

    counter = 0
    backup = 0
    for epoch in range(prev_epoch, num_epochs):

        running_loss = 0
        run = 0

        for i, train_data in enumerate(trainloader):
            if run == 10000:
               break
            inputs, labels = train_data
            x_args = tokenizer(list(inputs),return_tensors='pt',padding=True).to(device)
            y_args = tokenizer(list(labels),return_tensors='pt',padding=True).to(device)

            x_input_ids, x_token_type_ids, x_attention_mask = x_args['input_ids'], x_args['token_type_ids'], x_args['attention_mask']
            y_input_ids, y_token_type_ids, y_attention_mask = y_args['input_ids'], y_args['token_type_ids'], y_args['attention_mask']
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            outputs=model(x_input_ids, x_token_type_ids, x_attention_mask, y_input_ids[:,:-1], y_token_type_ids[:,:-1], y_attention_mask[:,:-1])
            loss=criterion(outputs.reshape(-1, outputs.shape[2]), y_input_ids[:, 1:513].reshape(-1))
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            running_loss += loss.item()
            run += 1

        print(f'epoch {epoch+1}, training loss = {running_loss/(i+1)}')
        train_loss.append(running_loss/(i+1))

        running_loss = 0
        run = 0
        for i, val_data in enumerate(validloader):
            if run == 1000:
              break
            inputs, labels = val_data

            x_args = tokenizer(list(inputs),return_tensors='pt',padding=True).to(device)
            y_args = tokenizer(list(labels),return_tensors='pt',padding=True).to(device)

            x_input_ids, x_token_type_ids, x_attention_mask = x_args['input_ids'], x_args['token_type_ids'], x_args['attention_mask']
            y_input_ids, y_token_type_ids, y_attention_mask = y_args['input_ids'], y_args['token_type_ids'], y_args['attention_mask']

            outputs=model(x_input_ids, x_token_type_ids, x_attention_mask, y_input_ids[:,:-1], y_token_type_ids[:,:-1], y_attention_mask[:,:-1])
            loss=criterion(outputs.reshape(-1, outputs.shape[2]), y_input_ids[:, 1:513].reshape(-1))
            running_loss += loss.item()  
            run += 1

        # Save the best model
        if running_loss/(i+1) < min_valid_loss:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = True, save model')  
            torch.save(model.state_dict(), f'{model_name}.pt')    
            min_valid_loss = running_loss/(i+1)
        else:
            print(f'epoch {epoch+1}, validation loss = {running_loss/(i+1)}, lowest validation loss = False, do not save model')
        valid_loss.append(running_loss/(i+1))
        
        counter += 1

        # Regularly save models between save_epoch epochs, for resuming training
        if counter == save_epoch:
            counter = 0
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'training_loss': train_loss,
            'validation_loss': valid_loss,
            'min_valid_loss': min_valid_loss
            }, f'{checkpoint_path}_{backup}')
            print(f'Model checkpoint has been saved to {checkpoint_path}_{backup}')
            if backup == 0:
                backup = 1
            else:
                backup = 0
        with open(f'{model_name}_train_loss.txt', 'w') as f:
          for line in train_loss:
              f.write(str(line))
              f.write(' ')
        with open(f'{model_name}_val_loss.txt', 'w') as f:
          for line in valid_loss:
              f.write(str(line))
              f.write(' ')
        print('Loss has been updated.')

    return model, train_loss, valid_loss