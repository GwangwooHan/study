from re import L, S
import sys

from deep_time_series.core import MetricModule
sys.path.append('..')

import pytest

import logging
logger = logging.getLogger('test')

import numpy as np
import torch
import torch.nn as nn

from torchmetrics import MeanAbsoluteError,  MeanSquaredError

from deep_time_series import (
    BaseHead, Head, DistributionHead, MetricModule
)


def test_base_head_tag_assignment():
    head = BaseHead()
    head.tag = 'my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'

    head.tag = 'head.my_tag'

    assert head.tag == 'head.my_tag'
    assert head.label_tag == 'label.my_tag'

    with pytest.raises(TypeError):
        head.tag = 3

    head = BaseHead()
    with pytest.raises(TypeError):
        head.tag = nn.Module()


def test_base_head_loss_weight_assignment():
    head = BaseHead()
    assert abs(head.loss_weight - 1.0) < 1e-10

    head.loss_weight = 3.0
    assert abs(head.loss_weight - 3.0) < 1e-10

    with pytest.raises(TypeError):
        head.loss_weight = 'hello'

    with pytest.raises(TypeError):
        head.loss_weight = nn.Module()

    with pytest.raises(ValueError):
        head.loss_weight = -3


def test_metric_module():
    metric_module = MetricModule(
        tag='my_tag',
        metrics=[
            MeanAbsoluteError(),
            MeanSquaredError(),
        ]
    )

    assert metric_module.head_tag == 'head.my_tag'
    assert metric_module.label_tag == 'label.my_tag'

    outputs = {
        'head.my_tag': torch.zeros(size=(10, 5))
    }

    batch = {
        'label.my_tag': 2*torch.ones(size=(10, 5))
    }

    metric_module.update(outputs=outputs, batch=batch, stage='train')
    metrics = metric_module.compute(stage='train')

    assert torch.allclose(
        metrics['train/my_tag.MeanAbsoluteError'], torch.FloatTensor([2.0]))

    assert torch.allclose(
        metrics['train/my_tag.MeanSquaredError'], torch.FloatTensor([4.0]))

    metric_module.reset(stage='val')
    metrics = metric_module.compute(stage='val')

    assert torch.all(torch.isnan(metrics['val/my_tag.MeanAbsoluteError']))
    assert torch.all(torch.isnan(metrics['val/my_tag.MeanSquaredError']))

    metric_module.reset(stage='test')
    metrics = metric_module.compute(stage='test')

    assert torch.all(torch.isnan(metrics['test/my_tag.MeanAbsoluteError']))
    assert torch.all(torch.isnan(metrics['test/my_tag.MeanSquaredError']))

    metric_module.reset(stage='train')
    metrics = metric_module.compute(stage='train')

    assert torch.all(torch.isnan(metrics['train/my_tag.MeanAbsoluteError']))
    assert torch.all(torch.isnan(metrics['train/my_tag.MeanSquaredError']))

    head = BaseHead()
    head.tag = 'my_tag'

    assert not head.has_metrics

    head.metrics = [
        MeanAbsoluteError(),
        MeanSquaredError(),
    ]

    assert head.has_metrics

    logger.info(head.metrics(outputs, batch, stage='train'))


def test_head():
    head = Head(
        tag='my_tag',
        output_module=nn.Linear(5, 3),
        loss_fn=nn.MSELoss(),
        loss_weight=0.3,
        metrics=[MeanAbsoluteError(), MeanSquaredError()]
    )

    assert head.tag == 'head.my_tag'
    assert head.loss_weight == 0.3

    with pytest.raises(ValueError):
        head.loss_weight = -1.0

    x = torch.rand(2, 1, 5)
    y = head(x)

    assert y.shape == torch.Size([2, 1, 3])

    outputs = head.get_outputs()

    assert torch.allclose(outputs['head.my_tag'], y)

    head.reset()

    for i in range(10):
        y = head(x)

    outputs = head.get_outputs()

    assert outputs['head.my_tag'].shape == torch.Size([2, 10, 3])

    head.reset()
    with pytest.raises(RuntimeError):
        outputs = head.get_outputs()

    batch = {
        'label.my_tag': torch.zeros(size=(2, 10, 3))
    }

    head.reset()
    for i in range(10):
        y = head(x)

    outputs = head.get_outputs()

    loss = head.calculate_loss(outputs, batch)

    assert torch.allclose(loss, (outputs['head.my_tag']**2).mean())

    head.metrics.update(outputs, batch, stage='train')
    metrics = head.metrics.compute(stage='train')

    logger.info(metrics)


def test_distribution_head():
    head = DistributionHead(
        tag='my_tag',
        distribution=torch.distributions.Normal,
        in_features=5,
        out_features=3,
        loss_weight=0.3,
        metrics=[MeanAbsoluteError(), MeanSquaredError()],
    )

    assert head.tag == 'head.my_tag'
    assert head.loss_weight == 0.3

    with pytest.raises(ValueError):
        head.loss_weight = -1.0

    x = torch.rand(3, 1, 5)
    y = head(x)

    distributions = [
        torch.distributions.Exponential,
        torch.distributions.Bernoulli,
        torch.distributions.Beta,
        # torch.distributions.Binomial,
        torch.distributions.Categorical,
        torch.distributions.Cauchy,
        torch.distributions.Chi2,
        torch.distributions.ContinuousBernoulli,
        torch.distributions.Dirichlet,
        torch.distributions.Exponential,
        torch.distributions.FisherSnedecor,
        torch.distributions.Gamma,
        torch.distributions.Geometric,
        torch.distributions.Gumbel,
        torch.distributions.HalfCauchy,
        torch.distributions.HalfNormal,
        torch.distributions.Laplace,
        torch.distributions.LogNormal,
        torch.distributions.NegativeBinomial,
        torch.distributions.Normal,
        torch.distributions.OneHotCategorical,
        # torch.distributions.Pareto,
        torch.distributions.Poisson,
        torch.distributions.StudentT,
        # torch.distributions.Uniform,
        torch.distributions.VonMises,
        torch.distributions.Weibull,
        # torch.distributions.Wishart,
    ]

    x = torch.rand(3, 1, 5)
    for distribution in distributions:
        logger.debug(f'{distribution = }')

        head = DistributionHead(
            tag='my_tag',
            distribution=distribution,
            in_features=5,
            out_features=3,
            loss_weight=0.3,
        )

        y = head(x)

        if distribution is torch.distributions.Categorical:
            assert y.shape ==  torch.Size([3, 1])
        else:
            assert y.shape ==  torch.Size([3, 1, 3])

    head = DistributionHead(
        tag='my_tag',
        distribution=torch.distributions.OneHotCategorical,
        in_features=5,
        out_features=10,
        loss_weight=0.3,
    )

    logger.info(f'{head(x) = }')

    x = torch.rand(3, 1, 5)
    for distribution in distributions:
        logger.debug(f'{distribution = }')
        head = DistributionHead(
            tag='my_tag',
            distribution=distribution,
            in_features=5,
            out_features=3,
            loss_weight=0.3,
            metrics=[MeanAbsoluteError(), MeanSquaredError()],
        )

        for i in range(10):
            y = head(x)

        outputs = head.get_outputs()

        if distribution is torch.distributions.Categorical:
            assert outputs['head.my_tag'].shape ==  torch.Size([3, 10])

            batch = {
                'label.my_tag': torch.randint(0, 2, size=(3, 10))
            }

            loss = head.calculate_loss(outputs, batch)

            logger.info(f'{loss = }')
        else:
            assert outputs['head.my_tag'].shape ==  torch.Size([3, 10, 3])

            batch = {'label.my_tag': torch.rand(size=(3, 10, 3))}

            if distribution is torch.distributions.Bernoulli:
                label = (torch.rand(size=(3, 10, 3)) > 0.5).to(torch.float32)
                batch = {'label.my_tag': label}

            if distribution is torch.distributions.Dirichlet:
                label = torch.rand(size=(3, 10, 3))
                label = label / torch.sum(label, dim=2, keepdim=True)
                batch = {'label.my_tag': label}

            if distribution is torch.distributions.Geometric:
                label = torch.randint(1, 3, size=(3, 10, 3))
                batch = {'label.my_tag': label}

            if distribution is torch.distributions.NegativeBinomial:
                label = torch.randint(1, 3, size=(3, 10, 3))
                batch = {'label.my_tag': label}

            if distribution is torch.distributions.OneHotCategorical:
                label = nn.functional.one_hot(
                    torch.randint(1, 3, size=(3, 10)),
                    num_classes=3
                )

                batch = {'label.my_tag': label}

            if distribution is torch.distributions.Pareto:
                label = torch.rand(size=(3, 10, 3)) + 2.0
                batch = {'label.my_tag': label}

            if distribution is torch.distributions.Poisson:
                label = torch.randint(1, 3, size=(3, 10, 3))
                batch = {'label.my_tag': label}

            loss = head.calculate_loss(outputs, batch)

            logger.info(f'{loss = }')

        head.metrics.update(outputs, batch, stage='train')
        metrics = head.metrics.compute(stage='train')

        logger.info(metrics)