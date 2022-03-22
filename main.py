import torch
import torch.nn as nn
from data.WikiHow_Dataset import WikiHow_Dataset
from model.PointerBertSumAbs import PointerBertSumAbs
from train import train_PBSA
from transformers import BertTokenizerFast
import os

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 30
    learning_rate_encoder = 2e-3
    learning_rate_decoder = 1e-2
    batch_size = 2
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    tokenizer = BertTokenizerFast(os.path.join('data','bert-base-uncased-vocab.txt'))
    train_set = WikiHow_Dataset(os.path.join('data','train.csv'))
    val_set = WikiHow_Dataset(os.path.join('data','val.csv'))
    test_set = WikiHow_Dataset(os.path.join('data','test.csv'))

    PointerBertSumAbsmodel = PointerBertSumAbs(
        pointer=True, 
        BertModel='bert-base-uncased', 
        n_heads=8, 
        forward_expansion=1024, 
        dropout=0.1, 
        n_decoders=3,
        device=device
    )
    PointerBertSumAbsmodel, PointerBertSumAbsmodel_train_loss, PointerBertSumAbsmodel_valid_loss = train_PBSA(
        tokenizer=tokenizer, 
        model=PointerBertSumAbsmodel, 
        model_name='/content/gdrive/MyDrive/PointerBertSumAbs', 
        learning_rate_encoder=learning_rate_encoder,
        learning_rate_decoder=learning_rate_decoder,
        train_set=train_set, 
        val_set=val_set, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        device=device, 
        criterion=criterion, 
        checkpoint_path='/content/gdrive/MyDrive/PointerBertSumAbs_checkpoint.pt',
        resume_training=False, # Set this to True if you want to restore training
        save_epoch=1,
        last_backup=0
    )
    BertSumAbsmodel = PointerBertSumAbs(
        pointer=False, 
        BertModel='bert-base-uncased', 
        n_heads=8, 
        forward_expansion=1024, 
        dropout=0.1, 
        n_decoders=3,
        device=device
    )
    BertSumAbsmodel, BertSumAbsmodel_train_loss, BertSumAbsmodel_valid_loss = train_PBSA(
        tokenizer=tokenizer, 
        model=BertSumAbsmodel, 
        model_name='/content/gdrive/MyDrive/BertSumAbs', 
        learning_rate_encoder=learning_rate_encoder,
        learning_rate_decoder=learning_rate_decoder,
        train_set=train_set, 
        val_set=val_set, 
        batch_size=batch_size, 
        num_epochs=num_epochs, 
        device=device, 
        criterion=criterion, 
        checkpoint_path='/content/gdrive/MyDrive/BertSumAbs_checkpoint.pt',
        resume_training=False, # Set this to True if you want to restore training
        save_epoch=1,
        last_backup=0
    )

    # An example of loading saved model
    # BertSumAbsmodel = PointerBertSumAbs(
    # pointer=False, 
    # BertModel='bert-base-uncased', 
    # n_heads=8, 
    # forward_expansion=1024, 
    # dropout=0.1, 
    # n_decoders=3,
    # device='cuda'
    # )
    # BertSumAbsmodel.load_state_dict(torch.load('BertSumAbs.pt'))

    # An example of summarising text
    # BertSumAbsmodel = PointerBertSumAbs(
    #     pointer=False, 
    #     BertModel='bert-base-uncased', 
    #     n_heads=8, 
    #     forward_expansion=1024, 
    #     dropout=0.1, 
    #     n_decoders=3,
    #     device='cuda'
    # )
    # BertSumAbsmodel.to('cuda')
    # BertSumAbsmodel.predict(['Hello World', 'Machine Learning'], tokenizer)