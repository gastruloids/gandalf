"""Model training script."""

import json
import logging
import shutil
import argparse
import typing as tp
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from rich.progress import track
from sklearn.metrics import f1_score, roc_curve, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from turner_lab.models import UNet

logger = logging.getLogger(__name__)


def dice_score(gt: npt.NDArray, pred: npt.NDArray, smooth: float = 1.0) -> float:
    """Dice metric.

    Args:
        gt (npt.NDArray): Ground truth
        pred (npt.NDArray): Prediction
        smooth (float, optional): Smoothness parameter. Defaults to 1.0.

    Returns:
        float: Dice metric
    """
    intersection = np.sum(gt * pred).astype(float)
    return (2.0 * intersection + smooth) / (np.sum(gt) + np.sum(pred) + smooth)


class BinaryDiceLoss(torch.nn.Module):
    """Binary Dice Loss."""

    def __init__(self, smooth: float = 1.0, *args, **kwargs) -> None:
        """Initialize the BinaryDiceLoss.

        Args:
            smooth (float, optional): Smoothing of the loss function. Defaults to 1.0.
        """
        super().__init__(*args, **kwargs)
        self.smooth = smooth

    def forward(self, predict, target):
        """Forward pass of the loss function."""
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (predict * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predict.sum() + target.sum() + self.smooth
        )

        return 1 - dice


class RodsRingsDataset(Dataset):
    """PyTorch wrapper for the Rods and Rings dataset."""

    def __init__(self, image_list: tp.List[Path], num_images: int) -> None:
        """Initialize the RodsRingsDataset.

        Args:
            image_list (tp.List[Path]): List of image paths
        """
        super().__init__()
        self.image_list = image_list

        if num_images != 0:
            self.image_list = self.image_list[:num_images]

        self.images = self.build_images()

    def __len__(self) -> int:
        """Calculate the length of the dataset.

        Returns:
            int: Length of the dataset
        """
        return len(self.image_list)

    def __getitem__(self, index: int):
        """Get an item from the dataset.

        Args:
            index (int): Index to get
        """
        return self.images[index]

    def build_images(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """Build the images from the image list."""
        images = []

        for img_path in self.image_list:

            # Masks always have the same name as the image with png suffix
            rod_path = (
                img_path.parent.parent / "masks" / "rods" / img_path.name
            ).with_suffix(".png")

            ring_path = (
                img_path.parent.parent / "masks" / "rings" / img_path.name
            ).with_suffix(".png")

            img = Image.open(img_path)

            if img.mode == "I;16":
                # Images are 16 bit, convert to 8 bit
                img = np.array(img)
                img = img / 256
                img = Image.fromarray(img.astype(np.uint8))

            img = self.transform(img)

            rod_mask = self.transform(Image.open(rod_path))

            ring_mask = self.transform(Image.open(ring_path))

            images.append(
                {
                    "name": img_path.name,
                    "image": img,
                    "rod_mask": rod_mask,
                    "ring_mask": ring_mask,
                },
            )

        return images

    @staticmethod
    def transform(image: Image.Image) -> npt.NDArray:
        """Transform the image."""
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        return transform(image)


class Trainer:
    """Class to manage training and logging of the model."""

    def __init__(
        self, data_path: Path, output_path: Path, fold_no: int, num_images: int = 0
    ) -> None:
        """Initialise the Trainer."""
        self.data_path = data_path
        self.output_path = output_path
        self.num_images = num_images
        self.fold_no = fold_no

        self.iterations = 0
        self.best_metric = 0
        self.learning_rate = 1e-3
        self.max_iterations = 1e4
        self.batch_size = 8

        logger.info(
            f"""
            Initialising Trainer with:
            Data Path: {data_path}
            Output Path: {output_path}
            Fold No: {fold_no}

            Training Parameters:
            Learning Rate: {self.learning_rate}
            Max Iterations: {self.max_iterations}
            Batch Size: {self.batch_size}
        """
        )

    def run(self):
        """Run the training loop."""
        # Get the data
        self._make_loaders()

        # Run the training loop
        self.train()

        # Test the data
        self.test()

    def train(self):
        """Train the model."""
        self.writer = SummaryWriter(log_dir=self.output_path / "log")

        self.model = UNet(n_channels=1, n_classes=2)
        self.model.cuda()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.999
        )

        self.loss_func = BinaryDiceLoss()

        self.max_epoch = int(self.max_iterations // len(self.train_loader))

        for epoch in track(range(self.max_epoch), description="Training..."):

            self.train_epoch()

            self.scheduler.step()

            if epoch % 10 == 0:
                self.validate()

        # Save the model
        torch.save(self.model.state_dict(), self.output_path / "model.pth")

        self.writer.close()

    def train_epoch(self):
        """Train a single epoch."""
        self.model.train()

        for _, batch in enumerate(self.train_loader):
            self.train_batch(batch)

    def train_batch(self, batch: tp.Dict[str, torch.Tensor]):
        """Train a single batch."""
        self.optimizer.zero_grad()

        img = batch["image"].cuda()
        rod_mask = batch["rod_mask"].cuda()
        ring_mask = batch["ring_mask"].cuda()

        rod_pred, ring_pred = self.model(img)

        rod_loss = self.loss_func(rod_pred, rod_mask)
        ring_loss = self.loss_func(ring_pred, ring_mask)

        loss = rod_loss + ring_loss

        loss.backward()
        self.optimizer.step()

        self.writer.add_scalar("Loss", loss, self.iterations)

        self.iterations += 1

    def validate(self):
        """Validate the model."""
        self.model.eval()

        for _, batch in enumerate(self.test_loader):
            self.validate_batch(batch)

    def validate_batch(self, batch: tp.Dict[str, torch.Tensor]):
        """Validate a single batch."""
        with torch.no_grad():
            img = batch["image"].cuda()
            rod_mask = batch["rod_mask"].cuda()
            ring_mask = batch["ring_mask"].cuda()

            rod_pred, ring_pred = self.model(img)

            rod_loss = self.loss_func(rod_pred, rod_mask)
            ring_loss = self.loss_func(ring_pred, ring_mask)

            loss = rod_loss + ring_loss

            self.writer.add_scalar("Val Loss", loss, self.iterations)
            print(f"Val Loss: {loss}")

    def test(self):
        """Test the model."""
        self.model.eval()

        for _, batch in enumerate(self.test_loader):
            self.test_batch(batch)

    def test_batch(self, batch: tp.Dict[str, torch.Tensor]):
        """Test a single batch."""
        name = batch["name"]
        img = batch["image"].cuda()
        rod_mask = batch["rod_mask"].cuda()
        ring_mask = batch["ring_mask"].cuda()

        rod_pred, ring_pred = self.model(img)

        rod_loss = self.loss_func(rod_pred, rod_mask)
        ring_loss = self.loss_func(ring_pred, ring_mask)

        metrics = defaultdict(list)

        for idx in range(img.size(0)):

            i_name = name[idx]
            i_img = img[idx].cpu().data.numpy()

            i_rod_pred = rod_pred[idx].cpu().data.numpy()
            i_ring_pred = ring_pred[idx].cpu().data.numpy()
            i_rod_gt = rod_mask[idx].cpu().data.numpy()
            i_ring_gt = ring_mask[idx].cpu().data.numpy()

            # ROC Curves
            self._make_roc_curve(i_rod_pred, i_rod_gt, "Rod")
            self._make_roc_curve(i_ring_pred, i_ring_gt, "Ring")

            # Save fuzzy masks to file
            masks_path = self.output_path / "model" / "fuzzy_masks"

            if not masks_path.exists():
                masks_path.mkdir(parents=True, exist_ok=True)

            with open(masks_path / f"{i_name}_rod.npy", "wb") as f:
                np.save(f, i_rod_pred)

            with open(masks_path / f"{i_name}_ring.npy", "wb") as f:
                np.save(f, i_ring_pred)

            # Clip masks to give binary labels
            i_rod_gt = self._clip(i_rod_gt)
            i_ring_gt = self._clip(i_ring_gt)
            i_rod_pred = self._clip(i_rod_pred)
            i_ring_pred = self._clip(i_ring_pred)

            self._save_image(
                i_name, i_img, i_rod_pred, i_ring_pred, i_rod_gt, i_ring_gt
            )

            metrics["rod_dice"].append(dice_score(i_rod_pred, i_rod_gt))
            metrics["ring_dice"].append(dice_score(i_ring_pred, i_ring_gt))
            metrics["rod_f1"].append(f1_score(i_rod_pred.flatten(), i_rod_gt.flatten()))
            metrics["ring_f1"].append(
                f1_score(i_ring_pred.flatten(), i_ring_gt.flatten())
            )

        for name, metric in metrics.items():
            metrics[name] = np.mean(metric)

        # Save metrics as json
        metrics_path = self.output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            data = json.dumps(metrics)
            f.write(data)

    def _save_image(self, name, img, rod_pred, ring_pred, rod_gt, ring_gt):
        """Save the image and masks."""
        # Upscale all the images to the original image size
        img = img[0, :, :] * 255
        rod_pred = rod_pred * 255
        ring_pred = ring_pred * 255

        for folder_name in ["rods", "rings", "compare", "images"]:
            (self.output_path / "model" / folder_name).mkdir(
                parents=True, exist_ok=True
            )

        # Save the images
        Image.fromarray(img).convert("L").save(
            (self.output_path / "model" / "images" / name).with_suffix(".png")
        )
        Image.fromarray(rod_pred).convert("L").save(
            (self.output_path / "model" / "rods" / name).with_suffix(".png")
        )
        Image.fromarray(ring_pred).convert("L").save(
            (self.output_path / "model" / "rings" / name).with_suffix(".png")
        )

        plt.close()
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))

        ax[0][0].imshow(rod_pred, cmap="gray")
        ax[0][0].set_title("Rod Predictions")

        ax[1][0].imshow(rod_gt[0] * 255, cmap="gray")
        ax[1][0].set_title("Rod Ground Truth")

        ax[0][1].imshow(ring_pred, cmap="gray")
        ax[0][1].set_title("Ring Predictions")

        ax[1][1].imshow(ring_gt[0] * 255, cmap="gray")
        ax[1][1].set_title("Ring Ground Truth")

        plt.tight_layout()

        fig.savefig((self.output_path / "model" / "compare" / name).with_suffix(".png"))

    def _make_loaders(self) -> tp.Tuple[DataLoader, DataLoader]:
        """Make training and testing loaders."""
        logging.info(f"Loading data for fold {self.fold_no}")
        folds = list(self.data_path.glob("*"))

        train_folds = [
            fold for fold in folds if f"fold_{self.fold_no}" not in fold.name
        ]
        test_folds = [fold for fold in folds if f"fold_{self.fold_no}" in fold.name]

        train_list = []

        for fold in train_folds:
            train_list.extend(list(fold.glob("raw/*")))

        test_list = list(test_folds[0].glob("raw/*"))

        logging.info(f"Creating Dataset Objects...")
        train_dataset = RodsRingsDataset(train_list, self.num_images)
        test_dataset = RodsRingsDataset(test_list, 0)

        logging.info(f"Creating DataLoader Objects...")
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_loader = train_loader
        self.test_loader = test_loader

        return train_loader, test_loader

    def _clip(self, array: npt.NDArray):
        """Clip the confidence arrays to binary masks."""
        return (array > 0.5).astype(np.uint8)

    def _make_roc_curve(self, pred: npt.NDArray, gt: npt.NDArray, name: str):
        """Make an ROC curve."""
        plt.close()
        gt = self._clip(gt)
        fpr, tpr, _ = roc_curve(gt.flatten(), pred.flatten())

        plt.plot(fpr, tpr, label=name)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(
            f"ROC Curve for {name}. AUC: {roc_auc_score(gt.flatten(), pred.flatten())}"
        )

        plt.legend()

        plt.savefig(self.output_path / f"roc_curve_{name}.png")


def make_folds(num_folds: int = 5):
    """Make the folds for training."""
    logging.info(f"Creating {num_folds} folds")

    root_data = Path("./data")

    all_data = np.array(list(root_data.glob("raw/*")))

    np.random.shuffle(all_data)

    logging.info(f"Total data: {len(all_data)}")

    for idx, split in enumerate(list(np.array_split(all_data, num_folds))):

        logging.info(f"Creating fold {idx}")

        fold_path = root_data / "folds" / f"fold_{idx}"

        fold_path.mkdir(parents=True, exist_ok=True)

        for data in split:

            (fold_path / "raw").mkdir(parents=True, exist_ok=True)
            (fold_path / "masks" / "rods").mkdir(parents=True, exist_ok=True)
            (fold_path / "masks" / "rings").mkdir(parents=True, exist_ok=True)

            shutil.copy(data, fold_path / "raw" / data.name)

            rod_path = root_data / "masks" / "rods" / data.with_suffix(".png").name
            ring_path = root_data / "masks" / "rings" / data.with_suffix(".png").name

            shutil.copy(rod_path, fold_path / "masks" / "rods" / rod_path.name)
            shutil.copy(ring_path, fold_path / "masks" / "rings" / ring_path.name)

        logging.info(f"Fold {idx} created successfully.")

    logging.info("Folds created successfully.")


if __name__ == "__main__":

    if not Path("./data/folds").exists():
        make_folds()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold_no",
        type=int,
        default=0,
        help="Fold number to train on",
    )

    parser.add_argument(
        "--name",
        help="Name of master folder",
    )

    parser.add_argument(
        "--num_images", type=int, default=0, help="Number of images to train on"
    )

    args = parser.parse_args()

    root_output = Path(f"output_{args.name}")

    output_path = Path(root_output / f"fold_{args.fold_no}")

    data_path = Path("./data/folds")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=output_path / "log.log", level=logging.INFO)

    trainer = Trainer(
        data_path, output_path, fold_no=args.fold_no, num_images=args.num_images
    )
    trainer.run()
