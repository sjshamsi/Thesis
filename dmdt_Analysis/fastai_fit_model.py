def fit_model(df: pd.DataFrame, data_column_name: str, label_column_name: str,
              model, batch_size: int, validation_percentage: float):
    dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=lambda r: numpy_to_pil(r[data_column_name]),
    get_y=lambda r: r[label_column_name],
    splitter=RandomSplitter(valid_pct=validation_percentage),
    item_tfms=Resize(224)
    )a

    dls = dblock.dataloaders(df, bs=batch_size)
    dls.show_batch(max_n=9, ax=show_batch_ax)

    learn = vision_learner(dls, model, metrics=[accuracy, error_rate])
    lr_min = learn.lr_find(show_plot=True)
    
    learn.fine_tune(20, base_lr=lr_min.valley, cbs=[SaveModelCallback(monitor='valid_loss', comp=np.less, fname='equivalent_bins_individual_class_backgrounds_model'), EarlyStoppingCallback(monitor='valid_loss', patience=3)])
    
    learn.recorder.plot_loss()
    learn.load('equivalent_bins_individual_class_backgrounds_model')



    interp = ClassificationInterpretation.from_learner(learn)
    cm = interp.confusion_matrix()
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_percent, cmap='Blues', interpolation='nearest')

    # Add text annotations
    for i in range(len(cm_percent)):
        for j in range(len(cm_percent[i])):
            text = f"{cm_percent[i, j]:.1f}%\n({int(cm[i, j])})"
            textcolour = color = "white" if cm_percent[i, j] > 50 else "black"
            ax.text(j, i, text, ha="center", va="center", color="black")

    ax.set_title("Equivalent Bin Histograms minus Individual Class Backgrounds")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(interp.vocab)))
    ax.set_yticks(range(len(interp.vocab)))
    ax.set_xticklabels(interp.vocab)
    ax.set_yticklabels(interp.vocab)
    plt.colorbar(im)
    plt.show()

    preds, targets = learn.get_preds()

    pred_classes = preds.argmax(dim=1)

    # Generate the report
    report = classification_report(targets, pred_classes, target_names=dls.vocab)
    print(report)

    interp.plot_top_losses(k=6, figsize=(6, 6))  # Show 5 samples with largest loss

    interp.plot_top_losses(k=6, figsize=(6, 6), largest=False)  # Show 5 samples with lowest loss


print(f"Suggested Learning Rates: Min: {lr_min.valley}")

def plot_confusion_matrix(interp):
    cm = interp.confusion_matrix()
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm_percent, cmap='Blues', interpolation='nearest')

    # Add text annotations
    for i in range(len(cm_percent)):
        for j in range(len(cm_percent[i])):
            text = f"{cm_percent[i, j]:.1f}%\n({int(cm[i, j])})"
            textcolour = color = "white" if cm_percent[i, j] > 50 else "black"
            ax.text(j, i, text, ha="center", va="center", color="black")

    ax.set_title("Equivalent Bin Histograms minus Individual Class Backgrounds")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(len(interp.vocab)))
    ax.set_yticks(range(len(interp.vocab)))
    ax.set_xticklabels(interp.vocab)
    ax.set_yticklabels(interp.vocab)
    plt.colorbar(im)
    plt.show()

    preds, targets = learn.get_preds()

    pred_classes = preds.argmax(dim=1)

    # Generate the report
    report = classification_report(targets, pred_classes, target_names=dls.vocab)
    print(report)

    interp.plot_top_losses(k=6, figsize=(6, 6))  # Show 5 samples with largest loss

    interp.plot_top_losses(k=6, figsize=(6, 6), largest=False)  # Show 5 samples with lowest loss

    return learn