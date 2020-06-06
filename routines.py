import cv2
def create_x(x, cutting=False, blur=False):
        c, h, w = x.shape
        x = x.detach().numpy()
        x = x.transpose((1, 2, 0))
        if blur:
          karnel_size = np.random.randint(2, 4)
          #x = cv2.fromarray(x)
          x = cv2.blur(x,(karnel_size, karnel_size))

        if cutting:
          while True:
              deleted_area = np.random.uniform(0.05, 0.1) * h * w
              deleted_ratio = np.random.uniform(0.1, 0.5)
              new_w = int(np.sqrt(deleted_area / deleted_ratio))
              new_h = int(np.sqrt(deleted_area * deleted_ratio))
              left = np.random.randint(0, w)
              top = np.random.randint(0, h)

              if left +new_w <= w and top + new_h <= h:
                  break
          noise = np.random.uniform(1, 255, (new_h, new_w, c))
          #print(x[:, top:top + new_h, left:left + new_w].shape, noise.shape)
          x[top:top + new_h, left:left + new_w, :] = noise
          x = x.transpose((2, 0, 1))
        return torch.Tensor(x)


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def model_summary(model):
  print("model_summary")
  print()
  print("Layer_name"+"\t"*7+"Number of Parameters")
  print("="*100)
  model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
  layer_name = [child for child in model.children()]
  j = 0
  total_params = 0
  print("\t"*10)
  for i in layer_name:
    print()
    param = 0
    try:
      bias = (i.bias is not None)
    except:
      bias = False
    if not bias:
      param =model_parameters[j].numel()+model_parameters[j+1].numel()
      j = j+2
    else:
      param =model_parameters[j].numel()
      j = j+1
    print(str(i)+"\t"*3+str(param))
    total_params+=param
  print("="*100)
  print(f"Total Params:{total_params}")


def set_trainability(model, trainability=False):
  for param in model.parameters():
    param.requires_grad = trainability
