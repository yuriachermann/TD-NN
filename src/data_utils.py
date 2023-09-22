import os
import torch

def collectDatasetImages(data_path):
    ls_1, ls_2 = [], []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jpg"):
                ls_1.append(root + "/" + file)
            elif file.endswith(".json"):
                ls_2.append(root + "/" + file)

    image_paths = sorted(ls_1, key=lambda x: str(x.split("/")[-1].split(".")[0]))
    annot_paths = sorted(ls_2, key=lambda x: str(x.split("/")[-1].split(".")[0]))

    return image_paths, annot_paths

def collectImagePaths(path="./data/crop_data"):
    image_paths, root_list = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(file)
                root_list.append(root)

    for i in range(len(root_list)):
        image_paths[i] = root_list[i] + "/" + image_paths[i]

    print(f"A total of {len(image_paths)} images found in given path.")

    return image_paths

def load_state_from_path(model, model_path, load_training, state_load):
    if load_training:
        model.load_state_dict(torch.load(model_path + "pytorch_model.bin"))
        print("----- Loading Model State done! -----")
    # load state of model if given
    if state_load is not None:
        model.load_state_dict(state_load)
        print("----- Loading Model State done! -----")
    return model


def save_model(model, optimizer, epoch, stats, savepath=None):
    """ Saving model checkpoint """

    if(not os.path.exists("models")):
        os.makedirs("models")
    if savepath == None:
        savepath = f"models/checkpoint_{model.__class__.__name__}_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath, device):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath, map_location="cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    model = model.to(device)
    optimizer_to(optimizer, device)

    return model, optimizer, epoch, stats

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def load_model_by_params(model, optimizer, savepath, device):
    """ Loading pretrained checkpoint """

    checkpoint = torch.load(savepath, map_location=device)
    print(len(checkpoint['model_state_dict']))
    model_list = model.named_parameters()
    for (name, module), c in zip(model_list, checkpoint['model_state_dict']):
        print(name, c)

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint["epoch"]
    # stats = checkpoint["stats"]
    # model = model.to(device)
    # optimizer_to(optimizer, device)

