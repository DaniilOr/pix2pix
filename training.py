def train(generator, discriminator, dataloader, opt1, opt2, scheduler, n_epoches=100, l1_lambda=25):
  history = {}
  generator.to(device)
  discriminator.to(device)
  loss_bce = nn.BCELoss()
  loss_l1 = nn.L1Loss()
  history['generator_loss'] = []
  history['discriminator_loss'] = []
  for epoch in range(n_epoches):
    losses1 = []
    losses2 = []
    for batch in dataloader:
      running_loss_d = 0
      running_loss_g = 0
      X = batch['X']
      Y = batch['Y']
      X = Variable(X.to(device))
      Y = Variable(Y.to(device))
      discriminator.zero_grad()
      set_trainability(discriminator, True)
      set_trainability(generator, False)

      real_res = discriminator(X, Y)

      fake_img = generator(X)

      fake_res = discriminator(X, fake_img)


      REAL = Variable(torch.ones(real_res.size())).to(device)

      FAKE = Variable(torch.ones(fake_res.size())).to(device)


      real_val_loss = loss_bce(real_res, REAL)
      fake_val_loss = loss_bce(fake_res, FAKE)
      # эта модель проще, поэтому учится быстрее, надо ее замедлить
      running_loss_d = (real_val_loss + fake_val_loss) * 0.5
      running_loss_d.backward()
      opt2.step()
      # training generator
      set_trainability(discriminator, False)
      set_trainability(generator, True)

      generator.zero_grad()
      fake_img = generator(X)

      fake_res = discriminator(X, fake_img)
      similarity = loss_l1(fake_img, Y)
      fooling = loss_bce(fake_res, REAL)
      # здесь все же ошибку дескриминатора увеличим
      running_loss_g = l1_lambda * similarity + fooling * 4
      running_loss_g.backward()

      opt1.step()
      losses1.append(running_loss_g)
      losses2.append(running_loss_d)
      torch.cuda.empty_cache()

    scheduler.step()
    clear_output(wait=True)
    rnd_ind = randint(0, len(fake_img)-1)
    print('Epoch: %d, generator loss :%.3f, discriminator loss: %.3f' % (epoch, running_loss_g ,running_loss_d))
    img_to_show = np.array(Y[rnd_ind].detach().cpu().numpy()).clip(0,1)
    fig=plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img_to_show.transpose((1, 2, 0)))
    img_to_show = np.array(fake_img[rnd_ind].detach().cpu().numpy()).clip(0,1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img_to_show.transpose((1, 2, 0)))
    plt.show(block=True)

    history['generator_loss'].append(sum(losses1)/len(losses1))
    history['discriminator_loss'].append(sum(losses2)/len(losses2))

  return history
