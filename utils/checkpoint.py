import torch

def save_checkpoint( model,optimizer,epoch,val_loss,val_psnr,val_ssim,path,):
    """
    保存训练状态。
    """
    checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),"optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss, "val_psnr": val_psnr, "val_ssim": val_ssim,
    }

    torch.save(checkpoint,path)