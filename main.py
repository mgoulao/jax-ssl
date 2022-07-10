import argparse
import jax
import haiku as hk
import jax.numpy as jnp
import optax
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
import time
import utils


def create_args_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--checkpoints_dir", default=".", type=str)
    parser.add_argument("--checkpoint_file", default=None, type=str)
    parser.add_argument("--base_lr", default=0.2, type=float)
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args):
    print(f"Using: {jax.devices()[0]}")
    rng_key = jax.random.PRNGKey(args.seed)

    normalize = T.Compose(
        [
            JaxNumpyCast(),
        ]
    )
    augmentation1 = [
        T.RandomResizedCrop(84, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur((7, 7), sigma=(0.1, 0.2))], p=1.0),
        T.RandomHorizontalFlip(),
        normalize,
    ]
    augmentation2 = [
        T.RandomResizedCrop(84, scale=(0.2, 1.0)),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur((7, 7), sigma=(0.1, 0.2))], p=0.1),
        T.RandomSolarize(120, p=0.2),
        T.RandomHorizontalFlip(),
        normalize,
    ]

    dataset = MNIST(
        ".",
        download=True,
        transform=SiameseAugmentation(
            T.Compose(augmentation1), T.Compose(augmentation2)
        ),
    )
    train_dataloader = NumpyLoader(dataset, num_workers=0, batch_size=64)

    model = hk.transform_with_state(_vicreg)
    model_init = model.init
    model_forward = model.apply
    model_forward = jax.jit(model_forward)

    model_params, model_state = model_init(
        rng_key,
        jax.random.uniform(rng_key, dataset[0][0][0].shape, jnp.float32, 0, 1),
        jax.random.uniform(rng_key, dataset[0][0][0].shape, jnp.float32, 0, 1),
    )

    if args.checkpoint_file is not None:
        model_params, model_state = utils.load_model(args)

    train(args, train_dataloader, model_forward, model_params, model_state, rng_key)


@optax.inject_hyperparams
def lars_optimiser(lr_schedule):
    return optax.lars(lr_schedule)


def train(args, train_loader, model_forward, model_params, model_state, rng_key):
    print("Starting Train")
    n_epochs = 10
    n_warmup_epochs = 2
    base_lr = args.base_lr
    lr_schedule = create_learning_rate_schedule(
        n_epochs, n_warmup_epochs, base_lr, len(train_loader)
    )
    optimiser = lars_optimiser(lr_schedule)
    optimiser_state = optimiser.init(model_params)

    for epoch in range(n_epochs):
        start_time = time.time()
        for i, (x, _) in enumerate(train_loader):
            x_ = x[1]
            x = x[0]

            (loss, model_state), grads = jax.value_and_grad(
                model_forward, has_aux=True
            )(model_params, model_state, rng_key, x, x_)
            if i % args.log_freq == 0:
                curr_lr = optimiser_state.hyperparams["lr_schedule"].item()
                print(
                    f"epoch: [{epoch}/{n_epochs}], iter: [{i}/{len(train_loader)}], time: {time.time()-start_time:.2f}, loss: {loss:.6f}, lr: {curr_lr:.6f}"
                )
            back_start_time = time.time()
            updates, optimiser_state = optimiser.update(
                grads, optimiser_state, model_params
            )
            model_params = optax.apply_updates(model_params, updates)

        utils.save_model(args, model_params, model_state, epoch)
    return model_params, model_state


def create_learning_rate_schedule(
    n_epochs, n_warmup_epochs, base_learning_rate, steps_per_epoch
):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=n_warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(n_epochs - n_warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[n_warmup_epochs * steps_per_epoch]
    )
    return schedule_fn


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].reshape(n - 1, n + 1)[:, 1:].flatten()


class VICReg:
    def __init__(self):
        self.num_features = 512
        self.encoder = hk.nets.ResNet(
            [2, 2, 2, 2],
            256,
            resnet_v2=True,
            bottleneck=True,
            logits_config={"w_init": None, "b_init": None, "name": "logits"},
        )
        self.predictor = hk.nets.MLP([512, self.num_features])
        self.path_1 = [self.encoder, self.predictor]
        self.path_2 = [self.encoder, self.predictor]
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, x: jnp.array, x_: jnp.array):
        if x.ndim < 4:
            x = jnp.expand_dims(x, 0)
            x_ = jnp.expand_dims(x_, 0)
        for layer in self.path_1:
            if isinstance(layer, hk.nets.ResNet):
                x = layer(x, self.training)
            else:
                x = layer(x)

        for layer in self.path_2:
            if isinstance(layer, hk.nets.ResNet):
                x_ = layer(x_, self.training)
            else:
                x_ = layer(x_)

        loss = self.loss(x, x_)
        return loss

    def cov_loss(self, x, x_):
        cov_x = (x.T @ x) / jnp.maximum(1, x.shape[0] - 1)
        cov_x_ = (x_.T @ x_) / jnp.maximum(1, x_.shape[0] - 1)
        off_diagonal_x = off_diagonal(cov_x)
        off_diagonal_x_ = off_diagonal(cov_x_)

        return jnp.divide(
            jnp.power(off_diagonal_x, 2).sum(), self.num_features
        ) + jnp.divide(jnp.power(off_diagonal_x_, 2).sum(), self.num_features)

    def std_loss(self, x, x_):
        gamma = 1
        std_y = jnp.sqrt(x.var(axis=0) + 0.0001)
        std_y_ = jnp.sqrt(x_.var(axis=0) + 0.0001)

        return (
            jnp.mean(jax.nn.relu(gamma - std_y)) / 2
            + jnp.mean(jax.nn.relu(gamma - std_y_)) / 2
        )

    def loss(self, x: jnp.array, x_: jnp.array):
        invariance_ratio = 10.0
        variance_ratio = 10.0
        covariance_ratio = 1.0
        invariance = jax.vmap(optax.l2_loss)(x_, x).mean()

        x = x - x.mean(axis=0)
        x_ = x_ - x_.mean(axis=0)
        variance = self.std_loss(x, x_)
        covariance = self.cov_loss(x, x_)

        return (
            invariance_ratio * invariance
            + variance_ratio * variance
            + covariance_ratio * covariance
        )


def _vicreg(x, x_):
    return VICReg()(x, x_)


class SiameseAugmentation:
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class JaxNumpyCast(object):
    def __call__(self, pic):
        return np.expand_dims(np.array(pic, dtype=jnp.float32) / 255.0, 0)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


if __name__ == "__main__":
    parser = create_args_parser()
    args = parser.parse_args()

    if args.debug:
        jax.disable_jit()

    main(args)
