# Model training

Add **Regression Model** or **Classification Model** from the node menu.
The node and its Train card share a numbered name, for example
`Classification Model 1`. Use **Show node** on the card to find it on the canvas.

## Inputs and architecture

The six inputs are paired: Train Features/Target, Validation Features/Target,
and Test Features/Target. Train is required. Optional splits must have both
inputs connected. The Train button appears when these connection rules hold.

Features must be numeric DataFrames without missing or infinite values. Each
Target must be a Series with the same row count and index as its Features.
Optional feature columns must match Train; they are reordered to Train's order.
Apply encoding and other preprocessing before the model node.

Add Linear and activation layers in execution order. The first Linear input
size must match the number of features; consecutive dimensions must agree.
Regression requires one output. Classification requires one output per Train
class and a final Linear layer with raw logits. Do not append Softmax or Sigmoid
to a classifier trained with CrossEntropy. Classification labels can be strings
or numbers; the label mapping is fitted on Train only. An unseen class in
Validation/Test produces an error.

## Training controls

Each card has its own learning rate and step count. **One step is one epoch**
(a full pass over Train in shuffled batches of up to 256 rows). Training uses
Adam on CPU in a background thread. Train All starts eligible nodes using their
individual settings and opens the Train tab.

Pause stops updates between batches. A partially completed epoch may already
have updated weights; only full epochs increment the completed-epoch counter.
Continue training retains weights and optimizer state and runs the configured
number of additional epochs. Changing LR affects the next run. Architecture
controls are disabled while training; changing the architecture after a pause
discards the old session and starts a new model on the next Train action.
Input data is copied when a run starts; later input changes apply to the next run.

The loss chart shows recent updates, and the table shows metrics over **all
rows** of every connected split. Only Train participates in gradient updates.
Regression uses MSE loss, RMSE, MAE and R2 (undefined for a single sample).
Classification uses CrossEntropy loss, Accuracy and macro-averaged Precision,
Recall and F1. Undefined class scores are reported as zero.

## Export and import

Export is available while paused or after completion. A ZIP contains:

- `metadata.json`: format version, task, architecture, feature schema, class
  mapping, completed epochs, settings and latest metrics.
- `weights.pt`: PyTorch model `state_dict` and Adam state.
- `model.py`: a standalone `create_model()` template for the architecture.

Use **Import model** in a node of the matching task. The archive restores the
layer controls, weights and training settings without entering them again.
Connect datasets with the same feature names/order and compatible class labels
to continue training. Datasets themselves are not included in the archive.
Import uses `torch.load(..., weights_only=True)` on CPU and reconstructs only
supported layers from JSON; it does not execute the template from the archive.

Models are held in memory until the application closes. Export them to keep
them between sessions. Closing the window requests a pause and waits for active
workers to finish before exiting. Removing a node also requests a pause; its
card remains available for export.
