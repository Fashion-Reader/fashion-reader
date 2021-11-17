"""
"""

from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from loss import get_loss
from dataset import MyDataset
from metrics import get_accuracy_all


def prepare_dataloader(CFG, df, tokenizer):
    train_index, valid_index = train_test_split(range(len(df)), test_size=0.2)

    train_ = df.iloc[train_index].reset_index(drop=True)
    valid_ = df.iloc[valid_index].reset_index(drop=True)
    
    train_ds = MyDataset(train_, tokenizer)
    valid_ds = MyDataset(valid_, tokenizer)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )
    val_loader = DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(CFG, epoch, model, optimizer, train_loader, scaler, scheduler=None):
    model.train()

    running_loss = None
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)
    for step, (inputs, labels) in pbar:
        inputs = tuple(inp.to(CFG.device) for inp in inputs)
        labels = labels.to(CFG.device).float()

        with autocast():
            outputs = model(inputs)
            loss = get_loss(CFG, outputs, labels)
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad() 

            description = f'epoch {epoch} loss: {running_loss:.4f}'
            pbar.set_description(description)

    if scheduler is not None:
        scheduler.step()


def valid_one_epoch_accuracy(CFG, epoch, model, val_loader):
    model.eval()

    sample_num = 0
    loss_sum = 0
    accuracy_dict = {"cate_accuracy": 0,
                    "style_accuracy": 0,
                    "neckline_accuracy": 0,
                    "pattern_accuracy": 0,
                    "button_accuracy": 0,
                    "lace_accuracy": 0,
                    "zipper_accuracy": 0,
                    "pocket_accuracy": 0,
                    "slit_accuracy": 0,
                    "buckle_accuracy": 0,
                    "sub_group_accuracy": 0,
                    "sub_category_accuracy": 0,
                    "sub_color_accuracy": 0,
                    }

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), position=0, leave=True)
    for step, (inputs, labels) in pbar:
        inputs = tuple(inp.to(CFG.device) for inp in inputs)
        labels = labels.to(CFG.device).float()
        
        outputs = model(inputs)
        loss = get_loss(CFG, outputs, labels)
        accuracy_all = get_accuracy_all(CFG, outputs, labels)
        
        loss_sum += loss.item()*labels.shape[0]
        sample_num += labels.shape[0]

        if ((step + 1) % 1 == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)

        for k, v in accuracy_all.items():
            accuracy_dict[k] += v*labels.shape[0]

    for k, v in accuracy_dict.items():
        accuracy_dict[k] /= sample_num

    print(f"""validation epoch : {epoch}
            cate_accuracy = {accuracy_dict['cate_accuracy']:.4f}
            style_accuracy = {accuracy_dict['style_accuracy']:.4f}
            neckline_accuracy = {accuracy_dict['neckline_accuracy']:.4f}
            pattern_accuracy = {accuracy_dict['pattern_accuracy']:.4f}
            button_accuracy = {accuracy_dict['button_accuracy']:.4f}
            lace_accuracy = {accuracy_dict['lace_accuracy']:.4f}
            zipper_accuracy = {accuracy_dict['zipper_accuracy']:.4f}
            pocket_accuracy = {accuracy_dict['pocket_accuracy']:.4f}
            slit_accuracy = {accuracy_dict['slit_accuracy']:.4f}
            buckle_accuracy = {accuracy_dict['buckle_accuracy']:.4f}
            sub_group_accuracy = {accuracy_dict['sub_group_accuracy']:.4f}
            sub_category_accuracy = {accuracy_dict['sub_category_accuracy']:.4f}
            sub_color_accuracy = {accuracy_dict['sub_color_accuracy']:.4f}""")
    return accuracy_dict
