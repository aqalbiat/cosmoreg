# Train on four development folds and validate on the remaining fold.
# Use standardized-target MSE, AdamW, and validation-based early stopping.
# Save the best model alongside scaling statistics and experiment settings.
# Keep the test group out of model selection.

def train_one_epoch(model, train_loader, optimizer, loss_fn):
    model.train()

    total_squared_error = 0.0
    total_target_values = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()

        predictions = model(batch_X)
        loss = loss_fn(predictions, batch_y)

        # Your part: calculate gradients.
        loss.backward()

        # Your part: update parameters.
        optimizer.step()

        total_squared_error += loss.item() * batch_y.numel()
        total_target_values += batch_y.numel()

    return total_squared_error / total_target_values
