import logging
import os
import sys
from typing import Sequence

import fire
import matplotlib
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import tqdm
from torch import nn
from torch import optim
from torch.distributions import Normal
import scipy.stats
import json
import glob
from pathlib import Path
import math
from copy import deepcopy

from kramersmoyal import km

import torchsde

# Wildcard import so that imported files find all classes
from latent_sde import *

from string import ascii_lowercase

current_marker = 0
markers = ["^", "v", "<", ">"]

def next_marker():
    global current_marker
    return_marker = current_marker
    current_marker = (current_marker + 1) % len(markers)
    return markers[return_marker]

def reset_marker():
    global current_marker
    current_marker = 0

interval_names = {
    "0_firsthalftrain": "$(0, 0.5{t_{train}})$",
    "1_secondhalftrain": "Training Set",
    "2_train": "$(0, t_{train})$",
    "3_doubletrain": "$(0, 2 t_{train})$",
    "4_fivetrain": "$(0, 5 t_{train})$",
    "5_extrapolation": "Test Set",
}

def draw_marginals(xs_sde, xs_data, file, title):
    bins = np.linspace(
        min(xs_sde.min(), xs_data.min()), max(xs_sde.max(), xs_data.max()), 100
    )
    plt.hist(
        torch.flatten(xs_sde).numpy(),
        bins=bins,
        alpha=0.35,
        label="Latent SDE",
        edgecolor="black",
        color="darkblue",
        linewidth=0.2,
    )
    plt.hist(
        torch.flatten(xs_data).numpy(),
        bins=bins,
        alpha=0.35,
        label="Data",
        edgecolor="black",
        color="orange",
        linewidth=0.2,
    )
    plt.legend()
    # plt.title(f"Marginals, {title}")
    plt.xlabel("Value $u(t)$")
    plt.ylabel("Frequency")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def distance_between_histograms(xs_sde, xs_data):
    max_val = 1000.0
    min_val = -1000.0
    constrain = lambda tens: torch.minimum(torch.maximum(tens, torch.full_like(tens, min_val)), torch.full_like(tens, max_val))
    values_sde = constrain(torch.flatten(xs_sde)).numpy()
    values_data = constrain(torch.flatten(xs_data)).numpy()

    return scipy.stats.wasserstein_distance(values_sde, values_data)


def mean(xs):
    return torch.mean(xs, dim=(1, 2))


def std(xs):
    return torch.std(xs, dim=(1, 2))


def bifurcation(xs):
    flattened_xs = xs.flatten()
    search_space = np.linspace(-0.5, 0.5, num=100)
    histogram, _ = np.histogram(flattened_xs, bins=search_space)
    min_point = np.argmin(histogram)
    if min_point == 0 or min_point == len(search_space) - 1:
        print(f"Warning: min_point={min_point}")

    return search_space[min_point]


def draw_xs(ts, xs, file, title, save=True):
    plt.plot(ts, xs, label="Data", linewidth=0.5)
    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")

    # plt.ylim(-4, 4)

    # plt.title(title)
    if save:
        plt.tight_layout(pad=0.3)
        plt.savefig(file + extension)
        plt.close()


def draw_mean_var(ts, xs_sde, xs_data, file, title):
    mean_sde = mean(xs_sde)
    conf_sde = std(xs_sde) * 1.96

    mean_data = mean(xs_data)
    conf_data = std(xs_data) * 1.96

    fig, ax = plt.subplots()
    ax.plot(ts, mean_sde, label="Latent SDE", color="darkblue")
    ax.fill_between(
        ts, (mean_sde - conf_sde), (mean_sde + conf_sde), color="darkblue", alpha=0.1
    )

    ax.plot(ts, mean_data, label="Data", color="orange")
    ax.fill_between(
        ts, (mean_data - conf_data), (mean_data + conf_data), color="orange", alpha=0.1
    )

    ax.legend()
    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")
    # plt.title(f"95% confidence, {title}")

    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def draw_posterior_around_data(ts, xs_posterior, xs_datapoint, file, title):
    fig, ax = plt.subplots()
    mean_posterior = mean(xs_posterior)
    conf_posterior = std(xs_posterior) * 1.96

    ax.plot(ts, mean_posterior, label="Posterior", color="darkblue")
    ax.fill_between(
        ts,
        (mean_posterior - conf_posterior),
        (mean_posterior + conf_posterior),
        color="darkblue",
        alpha=0.1,
    )
    ax.plot(ts, xs_datapoint[:, 0, 0], label="Data", color="orange", linewidth=2.0)

    ax.legend()

    plt.xlabel("Time $t$")
    plt.ylabel("Value $u(t)$")
    # plt.title(f"Posterior around data, {title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()


def kramers_moyal(ts, xs, dims=1):
    return kramers_moyal_one_dim(ts, xs)
    # if dims == 1:
    # elif dims == 2:
    #     return kramers_moyal_two_dims(ts, xs)

def kramers_moyal_one_dim(ts, xs):
    dt = ts[1] - ts[0]
    kmc_for_each_batch = []
    bin_space = (np.linspace(-4.0, 4.0, 500),)
    for batch_i in range(xs.size(dim=1)):
        current_timeseries = xs[:, batch_i, :]
        kmc, edges = km(np.nan_to_num(current_timeseries.numpy()), powers=2, bins=bin_space)
        kmc = kmc / dt
        kmc_for_each_batch.append(kmc)
    avg_kmc = sum(kmc_for_each_batch) / len(kmc_for_each_batch)
    return avg_kmc, bin_space[0]

def kramers_moyal_two_dims(ts, xs):
    dt = ts[1] - ts[0]
    kmc_for_each_batch = []
    bin_space = (np.linspace(-4.0, 4.0, 500),) * 2
    for batch_i in range(xs.size(dim=1)):
        current_timeseries = xs[:, batch_i, :]
        kmc, edges = km(np.nan_to_num(current_timeseries.numpy()), powers=np.array([[1, 0], [0, 1], [2, 0], [0, 2]]), bins=bin_space)
        kmc = kmc / dt
        kmc_for_each_batch.append(kmc)
    avg_kmc = sum(kmc_for_each_batch) / len(kmc_for_each_batch)
    return avg_kmc, bin_space

    raise NotImplementedError("Two-dimensional KM not implemented yet")

def draw_kramers_moyal(ts, xs_sde, xs_data, file, title, dims=1):
    km_data, bin_space1 = (
        kramers_moyal(ts, xs_data, dims=dims)
    )
    km_sde, bin_space2 = (
        kramers_moyal(ts, xs_sde, dims=dims)
    )

    if dims == 1:
        assert all(bin_space1 == bin_space2)

        num_subplots = km_data.size(dim=0)
        fig, axs = plt.subplots(num_subplots)
        fig.set_size_inches(3, 6)
        for i in range(1, num_subplots):
            axs[i - 1].plot(bin_space1[:-1], km_sde.numpy()[i, :], color="darkblue", label="Latent SDE")
            axs[i - 1].plot(bin_space2[:-1], km_data.numpy()[i, :], color="orange", label="Data")
            axs[i - 1].set_xlabel("y")
            axs[i - 1].set_ylabel(f"KM{i}")
        plt.legend()
        
        plt.title(f"KM factors, {title}")
        plt.tight_layout(pad=0.3)
        plt.savefig(file + extension)
        plt.close()
    # elif dims == 2:
    #     print(km_data.size())
    #     # torch.Size([5, 499, 499])
    #     print(km_sde.size())
    #     # torch.Size([5, 499, 499])
    #     # print(bin_space1)
    #     # bin_space1 == (np.linspace(-3.0, 3.0, 500),) * 2
    #     # print(bin_space2)
    #     # bin_space2 == (np.linspace(-3.0, 3.0, 500),) * 2
    #     num_subplots = km_data.size(dim=0)
    #     fig, axs = plt.subplots(num_subplots-1, 2, figsize=(12, 3 * num_subplots-1))

    #     km_titles = [
    #         "Drift $dx$",
    #         "Drift $dy$",
    #         "Diffusion $dx$",
    #         "Diffusion $dy$"
    #     ]
    #     for i in range(1, num_subplots):
    #         im1 = axs[i - 1, 0].imshow(km_sde.numpy()[i, :, :], aspect='auto', origin='lower', extent=[bin_space1[0][0], bin_space1[0][-1], bin_space1[0][0], bin_space1[0][-1]], cmap='viridis')
    #         axs[i - 1, 0].set_title(f"{km_titles[i-1]} Latent SDE")
    #         axs[i - 1, 0].set_xlabel("y1")
    #         axs[i - 1, 0].set_ylabel("y2")
    #         fig.colorbar(im1, ax=axs[i - 1, 0])
    #         im2 = axs[i - 1, 1].imshow(km_data.numpy()[i, :, :], aspect='auto', origin='lower', extent=[bin_space2[0][0], bin_space2[0][-1], bin_space2[0][0], bin_space2[0][-1]], cmap='viridis')
    #         axs[i - 1, 1].set_title(f"{km_titles[i-1]} Data")
    #         axs[i - 1, 1].set_xlabel("y1")
    #         axs[i - 1, 1].set_ylabel("y2")
    #         fig.colorbar(im2, ax=axs[i - 1, 1])

    #     plt.tight_layout(pad=0.3)
    #     plt.savefig(file + extension)
    #     plt.close()


# num_subplots = km_data.size(dim=0)
# fig, axs = plt.subplots(num_subplots-1, 2, figsize=(12, 3 * num_subplots-1))

# km_titles = [
#     "Drift $dx$",
#     "Drift $dy$",
#     "Diffusion $dx$",
#     "Diffusion $dy$"
# ]
# for i in range(1, num_subplots):
#     im1 = axs[i - 1, 0].imshow(km_sde.numpy()[i, :, :], aspect='auto', origin='lower', extent=[bin_space1[0][0], bin_space1[0][-1], bin_space1[0][0], bin_space1[0][-1]], cmap='viridis')
#     axs[i - 1, 0].set_title(f"{km_titles[i-1]} Latent SDE")
#     axs[i - 1, 0].set_xlabel("y1")
#     axs[i - 1, 0].set_ylabel("y2")
#     fig.colorbar(im1, ax=axs[i - 1, 0])
#     im2 = axs[i - 1, 1].imshow(km_data.numpy()[i, :, :], aspect='auto', origin='lower', extent=[bin_space2[0][0], bin_space2[0][-1], bin_space2[0][0], bin_space2[0][-1]], cmap='viridis')
#     axs[i - 1, 1].set_title(f"{km_titles[i-1]} Data")
#     axs[i - 1, 1].set_xlabel("y1")
#     axs[i - 1, 1].set_ylabel("y2")
#     fig.colorbar(im2, ax=axs[i - 1, 1])

# plt.tight_layout(pad=0.3)
# plt.savefig(file + extension)
# plt.close()


def tipping_rate(ts, xs):
    assert xs.size(dim=2) == 1

    tips_counted = torch.zeros_like(ts)

    bif = bifurcation(xs)

    xs_relative = (xs - bif)[:, :, 0]
    # now, positive == above bifurcation, negative == below
    cat_zeros = torch.zeros((1, xs_relative.size(dim=1)))
    xs_relative_1 = torch.cat([cat_zeros, xs_relative])
    xs_relative_2 = torch.cat([xs_relative, cat_zeros])
    xs_diff = xs_relative_1 * xs_relative_2
    tips_counted = torch.vmap(lambda x: x < 0)(xs_diff).sum(dim=1)[1:]

    # checked against this simpler, slower implementation :-)
    # for time in range(xs.size(dim=0) - 1):
    #     tips_counted_here = 0
    #     before = xs[time, :, 0]
    #     after = xs[time + 1, :, 0]
    #     for batch in range(xs.size(dim=1)):
    #         before = xs[time, batch, 0]
    #         after = xs[time + 1, batch, 0]
    #         if (before > bif and after <= bif) or (before < bif and after >= bif):
    #             tips_counted_here += 1
    #     tips_counted[time] = tips_counted_here

    # print(tips_counted == tips_counted_alternative)

    dt = ts[1] - ts[0]
    return tips_counted / (dt * (ts[-1] - ts[0]) * xs.size(dim=1))

def draw_tipping(ts, xs_sde, xs_data, window_size, file, title):
    tipping_data = (
        tipping_rate(ts, xs_data).unfold(0, window_size, window_size).mean(dim=1)
    )
    tipping_sde = (
        tipping_rate(ts, xs_sde).unfold(0, window_size, window_size).mean(dim=1)
    )

    plt.plot(ts[::window_size], tipping_sde, color="green", label="Latent SDE")
    plt.plot(ts[::window_size], tipping_data, color="orange", label="Data")
    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel("Tipping rate")
    # plt.title(f"Observed tips, {title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(file + extension)
    plt.close()



def explore_diffusion_balance(latent_sde_old, xs, ts, dt, beta, out):
    latent_sde = deepcopy(latent_sde_old)

    diffusion_sizes = list(map(lambda x: x / 20, list(range(100))))
    log_pxss = []
    log_ratios = []

    for diffusion_size in diffusion_sizes:
        def g(t, y):
            result =  torch.full_like(y, diffusion_size)
            return result
        latent_sde.g = g
        log_pxs, log_ratio, noise = latent_sde(
            xs, ts, 0.01, adjoint=False, method="euler", dt=dt
        )
        log_pxss.append(-float(log_pxs))
        log_ratios.append(beta * float(log_ratio))

    plt.plot(diffusion_sizes, log_pxss, color="darkblue", label="negative Log-Probability")
    plt.plot(diffusion_sizes, log_ratios, color="orange", label="$\\beta \\cdot$ KL Divergence")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Constant value of diffusion")
    plt.ylabel("Loss Score")
    plt.tight_layout(pad=0.3)
    plt.savefig(out + extension)
    plt.close()


def run_individual_analysis(model, data, training_info_file, config_file, show_params=False):
    out = model.replace(".pth", "")
    os.makedirs(out, exist_ok=True)
    print(f"Writing individual analysis to folder {out}")

    config = json.loads(Path(config_file).read_text())

    latent_sde = torch.load(model, map_location=torch.device("cpu"))
    if show_params:
        print(f"Parameters of model {model}:")
        for name, param in latent_sde.named_parameters():
            if param.requires_grad:
                print(name, param)
    tsxs_data = torch.load(data, map_location=torch.device("cpu"))

    ts_train = tsxs_data["ts_train"]
    ts_extrapolated = tsxs_data["ts_extrapolated"]

    dt = tsxs_data["dt"]

    batch_size = tsxs_data["xs"].size(dim=1)
    xs_sde_extrapolated = latent_sde.sample(batch_size, ts_extrapolated, dt=dt)
    xs_data_extrapolated = tsxs_data["xs"]

    print("??1")
    xs_sde_extrapolated_full = latent_sde.sample(batch_size, ts_extrapolated, dt=dt, project=False)
    fhn_gamma_extrapolated_full = FitzHughNagumoGamma().sample(batch_size, ts_extrapolated, False, "cpu", project=False)[0]

    print("??")
    datapoint_extrapolated = xs_data_extrapolated[:, 1:2, :]
    datapoint_extrapolated_repeated = datapoint_extrapolated.repeat(1, batch_size, 1)
    posterior_extrapolated, _, _ = latent_sde.posterior_plot(
        datapoint_extrapolated_repeated, ts_extrapolated, dt=dt
    )

    # just for the custom models because we did experiments on them, it's just
    # more convenient to plot that here even though it's not very general
    if latent_sde.pz0_mean.shape[1:][0] == 1 and "mean" in tsxs_data:
        ebm = StochasticEnergyBalance()
        mean = tsxs_data["mean"]
        std = tsxs_data["std"]
        print(mean, std)
        draw_func_ebm(latent_sde.h, ebm.f, f"{out}/func_drift", hardcoded_mean=mean, hardcoded_std=std)
        draw_func_ebm(latent_sde.g, ebm.g, f"{out}/func_diffusion", hardcoded_mean=mean, hardcoded_std=std)
    elif latent_sde.pz0_mean.shape[1:][0] == 2:
        t1 = float(ts_extrapolated[-1])
        draw_phase_portrait(latent_sde, t1, f"{out}/phase_portrait")
        draw_phase_portrait(FitzHughNagumo(), t1, f"{out}/phase_portrait_fhn")
        draw_phase_portrait(FitzHughNagumoGamma(), t1, f"{out}/phase_portrait_fhn_gamma")
        draw_phase_portrait(FitzHughNagumoKeno(), t1, f"{out}/phase_portrait_fhn_keno")

        draw_phase_portrait(latent_sde, t1, f"{out}/phase_portrait_diffusion", diffusion=True)
        draw_phase_portrait(FitzHughNagumo(), t1, f"{out}/phase_portrait_fhn_diffusion", diffusion=True)
        draw_phase_portrait(FitzHughNagumoGamma(), t1, f"{out}/phase_portrait_fhn_gamma_diffusion", diffusion=True)
        draw_phase_portrait(FitzHughNagumoKeno(), t1, f"{out}/phase_portrait_fhn_keno_diffusion", diffusion=True)
    

    # assumptions: ts_train[0] == 0, ts_train is evenly spaced
    assert ts_train[0] == 0.0

    intervals = {
        "0_firsthalftrain": (0, len(ts_train) // 2),
        "1_secondhalftrain": (len(ts_train) // 2, len(ts_train)),
        "2_train": (0, len(ts_train)),
        "3_doubletrain": (0, len(ts_train) * 2),
        "4_fivetrain": (0, len(ts_train) * 5),
        "5_extrapolation": (len(ts_train), len(ts_train) * 5),
    }

    info = {}

    for name, interval in intervals.items():
        title = interval_names[name]

        info_local = {}
        xs_data = tsxs_data["xs"][interval[0] : interval[1], :, :]
        xs_sde = xs_sde_extrapolated[interval[0] : interval[1], :, :]
        ts = ts_extrapolated[interval[0] : interval[1]]
        posterior = posterior_extrapolated[interval[0] : interval[1], :, :]
        datapoint = datapoint_extrapolated[interval[0] : interval[1], :, :]

        draw_marginals(xs_sde, xs_data, f"{out}/marginals_{name}", title)
        info_local["wasserstein_distance"] = distance_between_histograms(
            xs_sde, xs_data
        )

        draw_xs(ts, xs_sde[:, 0:20, 0], f"{out}/prior_{name}", f"Prior, {title}")
        draw_xs(ts, xs_data[:, 0:20, 0], f"{out}/data_{name}", f"Data, {title}")

        draw_mean_var(ts, xs_sde, xs_data, f"{out}/mean_var_{name}", title)

        draw_posterior_around_data(
            ts, posterior, datapoint, f"{out}/posterior_{name}", title
        )

        sde_dims = xs_sde_extrapolated_full.size(dim=2)

        # if sde_dims == 1:
        draw_kramers_moyal(ts, xs_sde, xs_data, f"{out}/km_{name}", title)
        # else:
        #     draw_kramers_moyal(ts, xs_sde_extrapolated_full[interval[0] : interval[1], :, :], fhn_gamma_extrapolated_full[interval[0] : interval[1], :, :], f"{out}/km_{name}", title, dims=1)
        draw_tipping(ts, xs_sde, xs_data, 5, f"{out}/tipping_{name}", title)

        if name == "2_train":
            try:
                explore_diffusion_balance(latent_sde, xs_data, ts, dt, config["beta"], f"{out}/diffusion_balance_{name}")
            except Exception as e:
                print(e)

        info_local["tipping_rate_data"] = float(tipping_rate(ts, xs_data).sum())
        info_local["tipping_rate_sde"] = float(tipping_rate(ts, xs_sde).sum())

        info_local["bifurcation_data"] = bifurcation(xs_data)
        info_local["bifurcation_sde"] = bifurcation(xs_sde)

        # if sde_dims == 1:
        km_sde, binspace_sde = kramers_moyal(ts, xs_sde)
        km_data, binspace_data = kramers_moyal(ts, xs_data)
        # else:
        #     km_sde, binspace_sde = kramers_moyal(ts, xs_sde_extrapolated_full[interval[0] : interval[1], :, :], dims=xs_sde_extrapolated_full.size(dim=2))
        #     km_data, binspace_data = kramers_moyal(ts, fhn_gamma_extrapolated_full[interval[0] : interval[1], :, :], dims=fhn_gamma_extrapolated_full.size(dim=2))
        assert all(binspace_data == binspace_sde)
        info_local["km_binspace"] = list(binspace_sde)
        info_local["km_sde_drift"] = list(km_sde.numpy()[1, :])
        info_local["km_sde_diffusion"] = list(km_sde.numpy()[2, :])
        info_local["km_data_drift"] = list(km_data.numpy()[1, :])
        info_local["km_data_diffusion"] = list(km_data.numpy()[2, :])

        info[name] = info_local

    # compute wasserstein distance for entire timeseries
    wasserstein_distances = []
    for interval in range(1, len(ts_extrapolated)):
        xs_data = tsxs_data["xs"][interval : interval + 1, :, :]
        xs_sde = xs_sde_extrapolated[interval : interval + 1, :, :]
        wasserstein_distances.append(distance_between_histograms(xs_sde, xs_data))
    plt.plot(ts_extrapolated[1:], wasserstein_distances)
    plt.xlabel("Time $t$")
    plt.xlabel("Wasserstein Distance")
    # plt.title("Wasserstein Distances")
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out}/wasserstein" + extension)
    plt.close()

    training_infos = json.loads(Path(training_info_file).read_text())
    for training_info in training_infos:
        draw_training_info(training_infos[training_info], training_info.capitalize(), out)

    with open(f"{out}/info.json", "w", encoding="utf8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def draw_param_to_info_both(
    configs,
    infos,
    ts,
    ts_title,
    param_name,
    param_title,
    info_name,
    info_title,
    out,
    xscale="linear",
):
    reset_marker()
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)
    # sort by the param, so first zip...
    tipping_rates_sde_sorted = sorted(
        zip(params, [x[ts][f"{info_name}_sde"] for x in infos])
    )
    tipping_rates_data_sorted = sorted(
        zip(params, [x[ts][f"{info_name}_data"] for x in infos])
    )
    # and then choose second item
    tipping_rates_sde = list(zip(*tipping_rates_sde_sorted))[1]
    tipping_rates_data = list(zip(*tipping_rates_data_sorted))[1]
    plt.scatter(sorted_params, tipping_rates_sde, label="Latent SDE", color="darkblue", marker=next_marker())
    plt.scatter(sorted_params, tipping_rates_data, label="Data", color="orange", marker=next_marker())
    plt.xlabel(param_title)
    plt.xscale(xscale)
    plt.ylabel(info_title)
    plt.legend()

    # plt.title(f"{param_title} to {info_title}, {ts_title}")
    plt.tight_layout(pad=0.3)
    plt.savefig(f"{out}/{info_name}_{param_name}_{ts}" + extension)
    plt.close()


def draw_param_to_info(
    configs,
    infos,
    ts,
    ts_title,
    param_name,
    param_title,
    info_name,
    info_title,
    out,
    xscale="linear",
    yscale="linear",
    save=True,
):
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)
    # sort by the param, so first zip...
    infos_sorted = sorted(zip(params, [x[ts][info_name] for x in infos]))
    # and then choose second item
    info_values = list(zip(*infos_sorted))[1]
    plt.scatter(sorted_params, info_values, label=ts_title, marker=next_marker())
    plt.xlabel(param_title)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.ylabel(info_title)
    if save:
        # plt.title(f"{param_title} to {info_title}, {ts_title}")
        plt.tight_layout(pad=0.3)
        plt.savefig(f"{out}/{info_name}_{param_name}_{ts}" + extension)
        plt.close()


def scatter_param_to_training_info(
    configs, training_infos, param_name, param_title, out, xscale="linear", tend=4
):
    reset_marker()
    params = [x[param_name] for x in configs]
    sorted_params = sorted(params)

    training_info_names = {
        "kl": ("KL Divergence", "darkblue"),
        "logpxs": ("Log-Likelihood", "orange"),
        "noise": ("Diffusion Size", "blue"),
    }

    for training_info_name, (
        training_info_title,
        color,
    ) in training_info_names.items():
        if training_info_name not in training_infos[0].keys():
            print(f"No {training_info_name} training info, skippping...")
            continue
        data = [x[training_info_name][-1] for x in training_infos]
        # diffusion size per time unit instead of integrated over the entire trajectory
        if training_info_title == "Diffusion Size":
            data = [p/tend for p in data]
        # sort by the param, so first zip...
        infos_sorted = sorted(
            zip(params, data)
        )
        # and then choose second item
        info_values = list(zip(*infos_sorted))[1]
        plt.scatter(sorted_params, info_values, color=color, marker=next_marker())
        # plt.title(f"{param_title} to {training_info_title}")
        plt.xlabel(param_title)
        plt.ylabel(training_info_title)
        plt.xscale(xscale)
        plt.tight_layout(pad=0.3)
        plt.savefig(
            f"{out}/training_info_{param_name}_{training_info_name}" + extension
        )
        plt.close()


def draw_training_info(training_info_xs, title, out, yscale="log"):
    plt.plot(training_info_xs)
    plt.xlabel("Iteration")
    plt.ylabel(title)
    plt.yscale(yscale)
    plt.tight_layout(pad=0.3)
    plt.savefig(
        f"{out}/training_info_{title}" + extension
    )
    plt.close()

def run_summary_analysis(model_folders, out):
    print(f"Writing summary analysis to folder {out}")
    # self-reported kwargs of simulations
    config_jsons = [os.path.join(x, "config.json") for x in model_folders]
    # summary statistics that we just generated
    info_jsons = [os.path.join(x, "model/info.json") for x in model_folders]
    # training info
    training_info_jsons = [os.path.join(x, "training_info.json") for x in model_folders]

    configs = [json.loads(Path(f).read_text()) for f in config_jsons]
    infos = [json.loads(Path(f).read_text()) for f in info_jsons]
    training_infos = [json.loads(Path(f).read_text()) for f in training_info_jsons]

    timespans = infos[0].keys()

    params = {
        "beta": ("$\\beta$", "log"),
        "context_size": ("Context Size", "linear"),
        "data_noise_level": ("Data Noise Level $\sigma$", "linear"),
        "noise_std": ("Noise Standard Deviation", "log"),
        "noise_penalty": ("Noise Penalty", "linear"),
        "batch_size": ("Size of Dataset", "log"),
    }

    for param_name, (param_title, xscale) in params.items():
        if not param_name in configs[0].keys():
            print(f"No {param_name}, skipping")
            continue
        params = [x[param_name] for x in configs]
        if None in params:
            print(f"None in {param_name}, skipping")
            continue
        if params.count(params[0]) == len(params):
            print(f"Only same value in {param_name}, skipping")
            continue

        for ts in timespans:
            draw_param_to_info_both(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "tipping_rate",
                "Tipping Rate",
                out,
                xscale=xscale,
            )

            draw_param_to_info_both(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "bifurcation",
                "Bifurcation",
                out,
                xscale=xscale,
            )

            draw_param_to_info(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "wasserstein_distance",
                "Wasserstein Distance",
                out,
                xscale=xscale,
            )

        # custom summary plots!

        old_figsize = plt.rcParams["figure.figsize"]
        plt.rcParams["figure.figsize"] = (5, 1.8)
        # custom wasserstein distance plot
        reset_marker()
        for ts in ["1_secondhalftrain", "5_extrapolation"]:
            draw_param_to_info(
                configs,
                infos,
                ts,
                interval_names[ts],
                param_name,
                param_title,
                "wasserstein_distance",
                "Wasserstein Distance",
                out,
                xscale=xscale,
                yscale="linear",
                save=False,
            )
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(pad=0.3)
        plt.savefig(f"{out}/custom_wasserstein" + extension)
        plt.close()

        # custom tipping rate plot
        reset_marker()
        for ts in ["1_secondhalftrain", "5_extrapolation"]:
            draw_param_to_info(
                configs,
                infos,
                ts,
                f"Latent SDE, {interval_names[ts]}",
                param_name,
                param_title,
                "tipping_rate_sde",
                "Tipping Rate",
                out,
                xscale=xscale,
                save=False,
            )
        for ts in ["1_secondhalftrain", "5_extrapolation"]:
            draw_param_to_info(
                configs,
                infos,
                ts,
                f"Data, {interval_names[ts]}",
                param_name,
                param_title,
                "tipping_rate_data",
                "Tipping Rate",
                out,
                xscale=xscale,
                save=False,
            )
        
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout(pad=0.3)

        plt.savefig(f"{out}/custom_tipping" + extension)
        plt.close()



        # Custom KM factors plot
        for ts in timespans:
            for param_value in params:
                # Plot that for these values
                chosen_param_values = [0, param_value]
                if not all([x in params for x in chosen_param_values]):
                    break
                chosen_param_indices = [params.index(x) for x in chosen_param_values]

                fig, axs = plt.subplots(1, 2)

                axs[0].plot(infos[0][ts]["km_binspace"][:-1], infos[chosen_param_indices[0]][ts]["km_data_drift"], label="Data")
                for i in chosen_param_indices:
                    axs[0].plot(infos[i][ts]["km_binspace"][:-1], infos[i][ts]["km_sde_drift"], label=f"$\\gamma = {params[i]}$")

                axs[1].plot(infos[0][ts]["km_binspace"][:-1], infos[chosen_param_indices[0]][ts]["km_data_diffusion"], label="Data")
                for i in chosen_param_indices:
                    axs[1].plot(infos[i][ts]["km_binspace"][:-1], infos[i][ts]["km_sde_diffusion"], label=f"$\\gamma = {params[i]}$")
                
                for ax in axs:
                    ax.set_xlabel("$x$")
                    ax.set_ylabel("$dx$")
                plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

                plt.tight_layout(pad=0.3)
                plt.savefig(f"{out}/kramersmoyal_{param_name}_{param_value}_{ts}" + extension)
                plt.close()

        plt.rcParams["figure.figsize"] = old_figsize
        scatter_param_to_training_info(
            configs, training_infos, param_name, param_title, out, xscale=xscale
        )

def draw_phase_portrait(sde, t1, out, diffusion=False):
    # t1 = t1 * 5
    batch_size = 1
    num_steps = 10000
    ts = torch.linspace(0, t1, steps=num_steps)
    if isinstance(sde, FitzHughNagumo) or isinstance(sde, FitzHughNagumoGamma) or isinstance(sde, FitzHughNagumoKeno):
        trajectories = sde.sample(batch_size, ts, False, "cpu", project=False)[0].numpy()
        if diffusion:
            sde_func = sde.g
        else:
            sde_func = sde.f
    else:
        trajectories = sde.sample(batch_size, ts, dt=t1/num_steps, project=False).numpy()
        if diffusion:
            sde_func = sde.g
        else:
            sde_func = sde.h

    for i in range(batch_size):
        start = int(num_steps*(4/5))
        plt.plot(trajectories[start:-1, i, 0:1], trajectories[start:-1, i, 1:2], linewidth=0.7, color="orange", alpha=1)

    xsize = plt.xlim()[1] - plt.xlim()[0]
    ysize = plt.ylim()[1] - plt.ylim()[0]
    scaleconst = 0.25
    y1 = np.linspace(plt.xlim()[0] - scaleconst * xsize, plt.xlim()[1] + scaleconst * xsize, 20)
    y2 = np.linspace(plt.ylim()[0] - scaleconst * ysize, plt.ylim()[1] + scaleconst * ysize, 20)

    # adapted from https://kitchingroup.cheme.cmu.edu/blog/2013/02/21/Phase-portraits-of-a-system-of-ODEs/
    g1, g2 = np.meshgrid(y1, y2)
    out1 = np.zeros(g1.shape)
    out2 = np.zeros(g2.shape)
    t = 0
    
    # https://stackoverflow.com/questions/24490753/logarithmic-lenghts-in-plotting-arrows-with-quiver-function-from-pyplot
    def transform(u, v):
        return u, v
        # arrow_lengths = np.sqrt(u*u + v*v)
        # len_adjust_factor = np.log10(arrow_lengths + 1) / arrow_lengths
        # return u*len_adjust_factor, v*len_adjust_factor

    for i in range(len(y1)):
        for j in range(len(y2)):
            x = g1[i, j]
            y = g2[i, j]
            dx, dy = map(float, sde_func(t, torch.tensor([[x, y]], dtype=torch.float32))[0])
            out1[i, j], out2[i, j] = transform(dx, dy)

    # plt.quiver(g1, g2, out1, out2, (out1**2 + out2**2)**0.5)
    plt.streamplot(g1, g2, out1, out2, color=(out1**2 + out2**2)**0.5, linewidth=0.4, cmap='viridis', density=1.2, arrowsize=0.4)

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout(pad=0.3)
    plt.savefig(out + extension)
    plt.close()

def draw_func_ebm(func_sde, func_ebm, out, hardcoded_mean=0.0, hardcoded_std=0.0):
    datapoints = 400
    space = np.linspace(-3, 3, datapoints)

    result_sde = []
    for x in space:
        result_sde.append(float(func_sde(0.0, torch.tensor([[x]], dtype=torch.float32))))
    result_ebm = []
    for x in space:
        result_ebm.append(float(func_ebm(0.0, torch.tensor([[x * hardcoded_std + hardcoded_mean]], dtype=torch.float32))) / hardcoded_std)

    plt.subplots()
    plt.axhline(0, linestyle='--', color="black")
    plt.plot(space, result_sde, label="Latent SDE", color="darkblue")
    plt.plot(space, result_ebm, label="Noisy EBM", color="orange")
    plt.xlabel("$x$")
    plt.ylabel("$dx$")
    plt.legend()
    plt.tight_layout(pad=0.3)
    plt.savefig(out + extension)
    plt.close()



def main(
    model=None, data=None, folder=None, pgf=False, only_summary=False, show_params=False, big=False
):
    global extension
    global plt
    if pgf:
        matplotlib.use("pgf")
        extension = ".pgf"
    else:
        extension = ".pdf"
    if big:
        extension = "_big" + extension
    import matplotlib.pyplot as plt

    if pgf:
        # from https://jwalton.info/Matplotlib-latex-PGF/
        plt.rcParams.update(
            {
                "font.family": "serif",  # use serif/main font for text elements
                "text.usetex": True,  # use inline math for ticks
                "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            }
        )
    plt.rcParams.update(
        {
            "figure.figsize": (5, 1.8) if big else (2.8, 1.8),
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.top": False,
            "ytick.right": False,
        }
    )

    # automatically walk through folder and find data.pth / model.pth pairs
    # also run analysis on entire benchmark

    models_and_data = []
    model_folders = None
    
    assert folder is not None

    model_files = glob.glob(f"{folder}/**/model.pth", recursive=True)
    model_folders = [os.path.dirname(x) for x in model_files]
    data_files = [os.path.join(x, "data.pth") for x in model_folders]
    assert all([os.path.exists(x) for x in data_files])
    training_info_files = [os.path.join(x, "training_info.json") for x in model_folders]
    assert all([os.path.exists(x) for x in training_info_files])
    config_files = [os.path.join(x, "config.json") for x in model_folders]
    assert all([os.path.exists(x) for x in config_files])
    models_and_data = list(zip(model_files, data_files, training_info_files, config_files))

    if not only_summary:
        for model, data, training_info_file, config_file in models_and_data:
            run_individual_analysis(model, data, training_info_file, config_file, show_params=show_params)

    # if we ran a batch analyze, run the meta-analysis as well
    if folder is not None:
        summary_folder = os.path.join(folder, "summary")
        os.makedirs(summary_folder, exist_ok=True)
        run_summary_analysis(model_folders, summary_folder)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    print(" ".join(sys.argv))
    torch.set_num_threads(2)
    fire.Fire(main)
