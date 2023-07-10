from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Callback
import numpy as np

from singer_identity.utils.core import similarity, roll
import singer_identity.losses as losses

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import random

# from evaluation_downstream.similarity.similarity_evaluator import EER_Eval_Dataset, EER


class BaseEvaluationCallback(Callback):
    """ """

    def __init__(
        self,
        log_n_epochs: int = 1,
        on_train: bool = False,
        use_projection: bool = True,
    ) -> None:
        """
        Args:

        """
        super().__init__()
        self.log_n_epochs = log_n_epochs
        self.on_train = on_train
        self.use_projection = use_projection

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
    ) -> None:
        if self.on_train:
            self.shared_step(trainer, pl_module, batch, batch_idx, "train")
            pl_module.train()

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.shared_step(trainer, pl_module, batch, batch_idx, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        self.shared_step(trainer, pl_module, batch, batch_idx, "")


class HypersphereEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        # if current_epoch % self.log_n_epochs == 0:

        pos_sample1 = batch["clip1"]
        pos_sample2 = batch["clip2"]
        pl_module.eval()
        if self.use_projection:
            z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
            name = "proj"
        else:
            z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
            name = "emb"

        align_loss = losses.align_loss(z1, z2)
        uniform_loss = losses.uniform_loss(z1) / 2
        uniform_loss += losses.uniform_loss(z2) / 2

        step_name = f"/{step_name}" if step_name != "" else step_name
        pl_module.log(
            f"Alignment evaluation {name}{step_name}",
            align_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=z1.shape[0],
        )
        pl_module.log(
            f"Uniformity evaluation {name}{step_name}",
            uniform_loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=z1.shape[0],
        )


class OrderEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_n_epochs == 0:
            pl_module.eval()
            pos_sample1 = batch["clip1"]
            pos_sample2 = batch["clip2"]
            labels = batch["group_name"]
            if self.use_projection:
                z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
                name = "proj"
            else:
                z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
                name = "emb"

            pair_positions = []
            times = 1
            for k in range(times):
                for i, inst_x in enumerate(z1):
                    # Find locations where label is different from anchor label
                    # And places current index at position 0
                    # valid_locations = [i] + [j for j,x in enumerate(labels) if x!=labels[i]]
                    valid_locations = [i] + [j for j, x in enumerate(labels) if i != j]

                    fit = (
                        similarity(inst_x[None, :], z2[valid_locations], 1)
                        .detach()
                        .cpu()
                        .numpy()
                    )  # .cpu().numpy()
                    order = np.argsort(fit)[::-1]
                    anchor_position = np.where(order == 0)[0][0]
                    pair_positions.append(anchor_position / len(valid_locations))
            mean = np.mean(pair_positions)
            var = np.var(pair_positions)
            median = np.median(pair_positions)
            step_name = f"/{step_name}" if step_name != "" else step_name
            pl_module.log(
                f"Order evaluation mean {name}{step_name}",
                mean,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )
            pl_module.log(
                f"Order evaluation median {name}{step_name}",
                median,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )
            pl_module.log(
                f"Order evaluation variance {name}{step_name}",
                var,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )


class EEREvaluation(BaseEvaluationCallback):
    """ """

    def __init__(
        self,
        *args,
        use_more_neg=False,
        **kwargs,
    ) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)
        self.use_more_neg = use_more_neg

    def separate_data(self, agg_vec):
        assert len(agg_vec) % 2 == 0
        total_num = len(agg_vec) // 2
        feature1 = agg_vec[:total_num]
        feature2 = agg_vec[total_num:]
        return feature1, feature2

    def eer(self, labels, scores):
        """
        labels: (N,1) value: 0,1 1 same 0 different
        scores: (N,1) value: -1 ~ 1
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        s = interp1d(fpr, tpr)
        a = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
        eer = brentq(a, 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)
        return (eer,)

    def find_matching_pairs(self, labels):
        """Only to be used for semi-supervised learning"""
        labels2 = list(labels).copy()
        idx = list(range(0, len(labels2)))

        labels2 = labels2[1:] + labels2[:1]
        idx = idx[1:] + idx[:1]

        to_fix_locations = [
            k for k, (a, b) in enumerate(zip(labels, labels2)) if a == b
        ]

        for i, loc in enumerate(to_fix_locations):  # locs where labels[i] == labels2[i]
            for j in range(0, len(labels2)):
                label_to_switch = labels2[j]
                label_to_change = labels2[loc]
                if (label_to_change != label_to_switch) and (
                    label_to_change != labels[j]
                ):
                    aux = labels2[j]
                    labels2[j] = labels2[loc]
                    labels2[loc] = aux

                    aux = idx[j]
                    idx[j] = idx[loc]
                    idx[loc] = aux
                    break
                elif j == (len(labels2) - 1):  # did not manage to find a pair
                    for k in range(0, len(labels2)):
                        if labels2[k] != labels2[loc]:
                            labels2[k] = labels2[loc]
                            idx[k] = idx[loc]

        # for i in range(0, len(labels2)):
        #     if labels[i] == labels2[i]:
        #         j = len(labels2) - 1
        #         while j > i:
        #             if labels2[i] != labels2[j]:  # finds a matching pair, switch i,j
        #                 aux = labels2[j]
        #                 labels2[j] = labels2[i]
        #                 labels2[i] = aux

        #                 aux = idx[j]
        #                 idx[j] = idx[i]
        #                 idx[i] = aux
        #                 break

        #             j -= 1

        return idx

    def shared_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        step_name: str,
    ):
        current_epoch = trainer.current_epoch

        if current_epoch % self.log_n_epochs == 0:
            pos_sample1 = batch["clip1"]
            pos_sample2 = batch["clip2"]
            labels = batch["group_name"]
            pl_module.eval()
            # z1 = pl_module(pos_sample1.to(pl_module.device))
            # z2 = pl_module(pos_sample2.to(pl_module.device))
            if self.use_projection:
                z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
                name = "proj"
            else:
                z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
                name = "emb"

            # positive scores
            scores = similarity(z1, z2).cpu().detach().tolist()
            labels = [1] * len(scores)
            # neg scores
            # if not self.use_more_neg:
            if True:
                # labels2 = labels.copy()
                neg = roll(z2)
                # idx = self.find_matching_pairs(group_labels)
                # neg = z2[idx]
            else:
                neg = pl_module(more_neg.to(pl_module.device))
            # Neg scores
            scores = scores + similarity(z1, neg).cpu().detach().tolist()
            labels = labels + ([0] * len(labels))

            eer, *others = self.eer(np.array(labels), np.array(scores))

            step_name = f"/{step_name}" if step_name != "" else step_name
            pl_module.log(
                f"EER evaluation {name}{step_name}",
                eer,
                on_epoch=True,
                sync_dist=True,
                batch_size=z1.shape[0],
            )


class KNNEvaluation(BaseEvaluationCallback):
    """ """

    def __init__(self, *args, n_neighbors=10, **kwargs) -> None:
        """
        Args:

        """
        super().__init__(*args, **kwargs)

        self.neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.le = {}
        self.vectors = {}
        self.labels = {}

    # Fit label encoders
    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for i, dataloader in enumerate(trainer.test_dataloaders):
            labels = list(trainer.test_dataloaders[i].dataset.groups.keys())
            self.le[i] = LabelEncoder()
            self.le[i].fit(labels)
            self.vectors[i] = []
            self.labels[i] = []

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: torch.Tensor,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        (pos_sample1, pos_sample2, more_neg, labels) = batch
        pl_module.eval()

        # z1 = pl_module(pos_sample1.to(pl_module.device))
        # z2 = pl_module(pos_sample2.to(pl_module.device))

        if self.use_projection:
            z1, z2 = (pl_module(pos_sample1), pl_module(pos_sample2))
            name = "proj"
        else:
            z1, z2 = (pl_module.encode(pos_sample1), pl_module.encode(pos_sample2))
            name = "emb"

        self.vectors[dataloader_idx].append(z1.cpu().numpy())
        self.labels[dataloader_idx].append(labels)

        if batch_idx == trainer.num_test_batches[dataloader_idx] - 1:
            X = np.concatenate([np.array(j) for j in self.vectors[dataloader_idx]])
            y = [k for j in self.labels[dataloader_idx] for k in j]
            y = self.le[dataloader_idx].transform(y)
            self.neigh.fit(X, y)
            score = self.neigh.score(X, y)
            pl_module.log(f"KNN score {name}", score)
            self.vectors[dataloader_idx] = []
            self.labels[dataloader_idx] = []


def sample_wavs_and_return_spk_pairs_list(groups, dev_ids, numbers):
    wav_list = []
    count_positive = 0
    print(f"generate {numbers} sample pairs")
    for _ in trange(numbers):
        prob = random.random()
        if prob > 0.5:
            dev_id_pair = random.sample(dev_ids, 2)

            # sample 2 wavs from different speaker
            sample1 = "/".join(random.choice(groups[dev_id_pair[0]]).split("/")[-2:])
            sample2 = "/".join(random.choice(groups[dev_id_pair[1]]).split("/")[-2:])

            label = "0"

            wav_list.append(",".join([label, sample1, sample2]))

        else:
            dev_id_pair = random.sample(dev_ids, 1)

            # sample 2 wavs from same speaker
            sample1 = "/".join(random.choice(groups[dev_id_pair[0]]).split("/")[-2:])
            sample2 = "/".join(random.choice(groups[dev_id_pair[0]]).split("/")[-2:])

            label = "1"
            count_positive += 1

            wav_list.append(",".join([label, sample1, sample2]))
    # print("finish, then dump file ..")
    # f = open(meta_data_name,"w")
    # for data in wav_list:
    #     f.write(data+"\n")
    # f.close()

    return wav_list
