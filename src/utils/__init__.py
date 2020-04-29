def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    
    return train_step
