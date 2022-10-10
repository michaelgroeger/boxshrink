import copy
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from config import (
    ACTIVATION,
    BATCH_SIZE,
    BEST_MODEL_DIR,
    CLASSES,
    DATA_DIR,
    DECODER,
    DEVICE,
    ENCODER,
    ENCODER_WEIGHTS,
    EVAL_ON_MASKS,
    EXPORT_BEST_MODEL,
    EXPORT_CSV_DIR,
    GAMMA,
    LEARNING_RATE,
    LEARNING_RATE_SCHEDULING,
    LOSS,
    MODE,
    N_EPOCHS,
    OPTIMIZER,
    PER_X_BATCH,
    PER_X_EPOCH,
    PER_X_EPOCH_PLOT,
    SCHEDULE_TYPE,
    START_EPOCH,
    STATE,
    STEP_SIZE,
    TRAINING_INPUT,
    WEIGHT_DECAY,
)
from dataset import Colonoscopy_Dataset
from tools import (
    epoch_time,
    human_sort,
    label_colors,
    return_batch_information,
    return_files_in_directory,
)
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex
from tqdm import tqdm

image_files = return_files_in_directory(DATA_DIR + "/original", ".tif")
mask_files = return_files_in_directory(DATA_DIR + "/ground_truth", ".tif")
box_files = return_files_in_directory(DATA_DIR + "/boxmasks", ".png")
rapid_masks = return_files_in_directory(DATA_DIR + "/testing/rapid_boxshrink", ".png")
robust_masks = return_files_in_directory(DATA_DIR + "/testing/robust_boxshrink", ".png")

human_sort(image_files)
human_sort(mask_files)
human_sort(box_files)
human_sort(rapid_masks)
human_sort(robust_masks)

from sklearn.model_selection import train_test_split

if TRAINING_INPUT == "boxes":
    X_train, X_test, y_train, y_test = train_test_split(
        image_files, box_files, test_size=0.1, random_state=1
    )
elif TRAINING_INPUT == "rapid_masks":
    X_train, X_test, y_train, y_test = train_test_split(
        image_files, rapid_masks, test_size=0.1, random_state=1
    )
elif TRAINING_INPUT == "robust_masks":
    X_train, X_test, y_train, y_test = train_test_split(
        image_files, robust_masks, test_size=0.1, random_state=1
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        image_files, mask_files, test_size=0.1, random_state=1
    )

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.11111, random_state=1
)  # 0.1111 x 0.9 = 0.1

if EVAL_ON_MASKS == True:
    if TRAINING_INPUT == "boxes":
        y_val = [
            i.replace("boxmasks", "ground_truth").replace("png", "tif") for i in y_val
        ]
        y_test = [
            i.replace("boxmasks", "ground_truth").replace("png", "tif") for i in y_test
        ]
    elif (
        TRAINING_INPUT == "rapid_masks" or TRAINING_INPUT == "CAM_sp_crf_masks_adjusted"
    ):
        y_val = [
            i.replace("/testing/rapid_boxshrink", "ground_truth").replace("png", "tif")
            for i in y_val
        ]
        y_test = [
            i.replace("/testing/rapid_boxshrink", "ground_truth").replace("png", "tif")
            for i in y_test
        ]
    elif TRAINING_INPUT == "robust_masks":
        y_val = [
            i.replace("/testing/robust_boxshrink", "ground_truth").replace("png", "tif")
            for i in y_val
        ]
        y_test = [
            i.replace("/testing/robust_boxshrink", "ground_truth").replace("png", "tif")
            for i in y_test
        ]

train_dataset = Colonoscopy_Dataset(X_train, y_train, limit_dataset_size=5)

test_dataset = Colonoscopy_Dataset(X_test, y_test, limit_dataset_size=5)

val_dataset = Colonoscopy_Dataset(X_val, y_val, limit_dataset_size=5)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# create segmentation model with pretrained encoder
if DECODER == "Unet":
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

elif DECODER == "DeepLabV3+":
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

if OPTIMIZER == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
if OPTIMIZER == "Adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduling
if LEARNING_RATE_SCHEDULING == True and SCHEDULE_TYPE == "STEP":
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA, verbose=True)
elif LEARNING_RATE_SCHEDULING == True and SCHEDULE_TYPE == "EXPONENTIAL":
    STEP_SIZE = "Not needed"
    scheduler = ExponentialLR(optimizer, gamma=GAMMA, verbose=True)
elif LEARNING_RATE_SCHEDULING == False:
    STEP_SIZE = "No scheduling"
    GAMMA = "No scheduling"
    SCHEDULE_TYPE = "No scheduling"

# Setup date for model name
from datetime import date

today = date.today()
datestring = today.strftime("%Y-%m-%d")
# Instantiate accuracy and loss tracker
best_train_loss = float("inf")
best_valid_loss = float("inf")
best_valid_iou = 0
best_train_iou = 0

# Build dataframe to collect loss and metric data
df_train = pd.DataFrame(columns=["epoch", "loss", "avg_loss", "mean_iou"])
df_val = pd.DataFrame(columns=["epoch", "loss", "avg_loss", "mean_iou"])
# # Determine column types train
# Dummy entry to prevent visualization bug that large values are plotted as zero
if LOSS == "CrossEntropyLoss":
    criterion = CrossEntropyLoss()
    criterion_double = CrossEntropyLoss()
jaccard = JaccardIndex(num_classes=len(CLASSES), reduction="elementwise_mean").to(
    DEVICE
)

torch.backends.cudnn.benchmark = True
train_start_time = time.time()
train_iou_score = torch.tensor([0])
early_stopped = 0
for epoch in range(START_EPOCH, N_EPOCHS):
    model.train()
    batch, running_epoch_iou, running_epoch_loss = 0, 0.0, 0.0
    with tqdm(train_loader, unit="batch") as tepoch:
        for train_inputs, train_labels, train_org_images in tepoch:
            batch += 1
            optimizer.zero_grad(set_to_none=True)
            tepoch.set_description(f"Epoch {epoch}")
            train_inputs, train_labels, train_org_images = (
                train_inputs.to(DEVICE),
                train_labels.to(DEVICE),
                train_org_images.to(DEVICE),
            )
            # forward
            train_outputs = model(train_inputs).to(DEVICE)
            out_max = (
                torch.argmax(train_outputs, dim=1, keepdim=True)[:, -1, :, :]
                .cpu()
                .detach()
                .numpy()
            )
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            if epoch % PER_X_EPOCH == 0 and batch % PER_X_BATCH == 0:
                return_batch_information(
                    train_org_images,
                    out_max,
                    train_labels,
                    1,
                    CLASSES,
                    label_colors=label_colors,
                )
            optimizer.step()
            model.eval()
            train_iou_score = jaccard(train_outputs, train_labels).to(DEVICE).item()
            model.train()
            running_epoch_iou += train_iou_score
            train_loss = float(train_loss.item())
            running_epoch_loss += train_loss
            # print statistics
            tepoch.set_postfix(
                phase="Training",
                loss=train_loss,
                iou=train_iou_score,
                epoch_iou=running_epoch_iou / batch,
                epoch_loss=running_epoch_loss / batch,
            )
        train_mean_epoch_iou, train_mean_epoch_loss = (
            running_epoch_iou / batch,
            running_epoch_loss / batch,
        )
    if best_train_loss > train_mean_epoch_loss:
        best_train_loss = train_mean_epoch_loss
    if best_train_iou < train_mean_epoch_iou:
        best_train_iou = train_mean_epoch_iou
    # Save results to dataframe
    if epoch == 0:
        train_row = {
            "epoch": int(epoch),
            "loss": float(train_mean_epoch_loss),
            "avg_loss": float(train_mean_epoch_loss),
            "mean_iou": float(train_mean_epoch_iou),
        }
    else:
        # Get moving average
        train_avg = df_train["loss"].ewm(com=0.99).mean()
        train_row = {
            "epoch": int(epoch),
            "loss": float(train_loss),
            "avg_loss": train_avg[(epoch - 1)],
            "mean_iou": train_mean_epoch_iou,
        }

    df_train = df_train.append(train_row, ignore_index=True)
    # Decay Learning Rate at x steps
    if LEARNING_RATE_SCHEDULING == True:
        scheduler.step()
    # Delete variables to free memory
    del running_epoch_iou, running_epoch_loss, train_loss, train_iou_score

    ### Running validation loop
    batch, running_epoch_iou, running_epoch_loss = 0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as tepoch:
            for val_inputs, val_labels, val_org_images in tepoch:
                batch += 1
                tepoch.set_description(f"Epoch {epoch}")
                val_inputs, val_labels, val_org_images = (
                    val_inputs.to(DEVICE),
                    val_labels.to(DEVICE),
                    val_org_images.to(DEVICE),
                )
                # forward
                val_outputs = model(val_inputs)
                # Collect metrics
                val_iou_score = jaccard(val_outputs, val_labels).item()
                val_loss = criterion(val_outputs, val_labels).item()
                # Collect data for dataframe
                running_epoch_iou += val_iou_score
                running_epoch_loss += val_loss
                # print statistics
                tepoch.set_postfix(
                    phase="Validation",
                    loss=val_loss,
                    iou=val_iou_score,
                    epoch_iou=running_epoch_iou / batch,
                    epoch_loss=running_epoch_loss / batch,
                )
        val_mean_epoch_iou, val_mean_epoch_loss = (
            running_epoch_iou / batch,
            running_epoch_loss / batch,
        )
        # Save results to dataframe
        if epoch == 0:
            val_row = {
                "epoch": int(epoch),
                "loss": float(val_mean_epoch_loss),
                "avg_loss": float(val_mean_epoch_loss),
                "mean_iou": val_mean_epoch_iou,
            }
        else:
            val_avg = df_val["loss"].ewm(com=0.99).mean()
            val_row = {
                "epoch": int(epoch),
                "loss": float(val_loss),
                "avg_loss": val_avg[(epoch - 1)],
                "mean_iou": val_mean_epoch_iou,
            }
        df_val = df_val.append(val_row, ignore_index=True)
        if best_valid_loss > val_mean_epoch_loss:
            # Update best metrics
            best_valid_loss = val_mean_epoch_loss
        if best_valid_iou < val_mean_epoch_iou:
            best_valid_iou = val_mean_epoch_iou
            best_model = copy.deepcopy(model)
            if epoch > 2 and EXPORT_BEST_MODEL == True:
                model_name = "_".join(
                    [
                        datestring,
                        STATE,
                        MODE,
                        DECODER,
                        OPTIMIZER,
                        LOSS,
                        ENCODER,
                        str(len(train_dataset)),
                        "images",
                        LOSS,
                        "loss",
                        str(best_valid_loss).replace(".", "_"),
                        "iou",
                        str(val_mean_epoch_iou),
                        "epoch",
                        str(epoch),
                        ".pth",
                    ]
                )
            # save model
            path = os.path.join(BEST_MODEL_DIR, model_name)
            torch.save(best_model.state_dict(), path)
            print(f"Model saved! Name is {model_name}")
        if epoch % PER_X_EPOCH_PLOT == 0:
            plt.plot(df_train["epoch"], df_train["avg_loss"], label="Train Loss")
            plt.plot(df_val["epoch"], df_val["avg_loss"], label="Valid Loss")
            plt.plot(df_val["epoch"], df_val["mean_iou"], label="Mean IoU")
            plt.legend()
            plt.title("Performance")
            plot = plt.gcf()
            plt.show()
        train_df_name = "_".join(
            [
                datestring,
                "train",
                MODE,
                DECODER,
                OPTIMIZER,
                LOSS,
                ENCODER,
                str(len(train_dataset)),
                "images",
                ".csv",
            ]
        )
        valid_df_name = "_".join(
            [
                datestring,
                "valid",
                MODE,
                DECODER,
                OPTIMIZER,
                LOSS,
                ENCODER,
                str(len(train_dataset)),
                "images",
                ".csv",
            ]
        )
        df_train.to_csv(os.path.join(EXPORT_CSV_DIR, train_df_name))
        df_val.to_csv(os.path.join(EXPORT_CSV_DIR, valid_df_name))
        if epoch > 5:
            last_runs = df_train["loss"][-5:]
            # Get min and max of that window
            min_loss_last_runs = last_runs.min()
            max_loss_last_runs = last_runs.max()
            difference = max_loss_last_runs - min_loss_last_runs
            if difference < 0.001:
                print("Stopped Training because it doesn't improve anymore.")
                train_end_time = time.time()
                # Get minutes and seconds to write to ML flow
                train_mins, train_secs = epoch_time(train_start_time, train_end_time)
                batch, running_epoch_iou, running_epoch_loss = 0, 0.0, 0.0
                best_model.eval()
                with torch.no_grad():
                    with tqdm(test_loader, unit="batch") as tepoch:
                        for test_inputs, test_labels, test_org_images in tepoch:
                            batch += 1
                            tepoch.set_description(f"Epoch {epoch}")
                            test_inputs, test_labels, test_org_images = (
                                test_inputs.to(DEVICE),
                                test_labels.to(DEVICE),
                                test_org_images.to(DEVICE),
                            )
                            # forward
                            test_outputs = best_model(test_inputs)
                            # Collect metrics
                            test_iou_score = jaccard(test_outputs, test_labels).item()
                            test_loss = criterion(test_outputs, test_labels).item()
                            # Collect data for dataframe
                            running_epoch_iou += test_iou_score
                            running_epoch_loss += test_loss
                            # print statistics
                            tepoch.set_postfix(
                                phase="Validation",
                                loss=test_loss,
                                iou=test_iou_score,
                                epoch_iou=running_epoch_iou / batch,
                                epoch_loss=running_epoch_loss / batch,
                            )
                    test_mean_epoch_iou, test_mean_epoch_loss = (
                        running_epoch_iou / batch,
                        running_epoch_loss / batch,
                    )
                print("")
                print(
                    f"Performance on test set: {val_mean_epoch_iou} IoU and {val_mean_epoch_loss} Loss"
                )
                print(f"Training time was {train_mins, train_secs}")
                early_stopped += 1
                break
if early_stopped == 0:
    train_end_time = time.time()
    # Get minutes and seconds to write to ML flow
    train_mins, train_secs = epoch_time(train_start_time, train_end_time)
    # End run and get status
    batch, running_epoch_iou, running_epoch_loss = 0, 0.0, 0.0
    best_model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for test_inputs, test_labels, test_org_images in tepoch:
                batch += 1
                tepoch.set_description(f"Epoch {epoch}")
                test_inputs, test_labels, test_org_images = (
                    test_inputs.to(DEVICE),
                    test_labels.to(DEVICE),
                    test_org_images.to(DEVICE),
                )
                # forward
                test_outputs = best_model(test_inputs)
                # Collect metrics
                test_iou_score = jaccard(test_outputs, test_labels).item()
                test_loss = criterion(test_outputs, test_labels).item()
                # Collect data for dataframe
                running_epoch_iou += test_iou_score
                running_epoch_loss += test_loss
                # print statistics
                tepoch.set_postfix(
                    phase="Validation",
                    loss=test_loss,
                    iou=test_iou_score,
                    epoch_iou=running_epoch_iou / batch,
                    epoch_loss=running_epoch_loss / batch,
                )
        test_mean_epoch_iou, test_mean_epoch_loss = (
            running_epoch_iou / batch,
            running_epoch_loss / batch,
        )
    print(f"Training time was {train_mins, train_secs}")
