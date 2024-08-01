import numpy as np

# Combinations of bias parameters for bispec kernels.
bispec_bias_lists = [
    [
        "c2_b2_f",
        "c2_b1_b2",
        "c2_b1_b1",
        "c2_b1_f",
        "c2_b1_f",
        "c1_b1_b1_f",
        "c1_b2_f",
        "c1_b1_b2",
        "c1_b1_b1",
        "c2_b1_b1_f",
        "c1_b1_f",
    ],
    ["c2_b1_f_f", "c1_f_f", "c1_f_f_f", "c2_f_f", "c2_f_f_f", "c1_b1_f_f"],
    [
        "c1_c1_f_f",
        "c2_c2_b1_f",
        "c2_c1_b1_f",
        "c2_c1_b1",
        "c2_c1_b2",
        "c2_c2_f_f",
        "c1_c1_f",
        "c2_c2_b1",
        "c2_c2_b2",
        "c2_c2_f",
        "c2_c1_b1_f",
        "c2_c1_f",
        "c1_c1_b1_f",
        "c1_c1_b1",
        "c1_c1_b2",
        "c1_c1_f_f",
    ],
    ["c1_c1_bG2", "c2_c2_bG2", "c2_c1_bG2"],
    ["c1_b1_bG2", "c1_bG2_f", "c2_bG2_f", "c2_b1_bG2"],
    [
        "b1_f_f",
        "b1_b1_f_f",
        "b1_b1_b2",
        "b2_f_f",
        "b1_b1_b1",
        "b1_b1_b1_f",
        "b1_b1_f",
        "b1_f_f_f",
        "f_f_f",
        "f_f_f_f",
        "b1_b2_f",
    ],
    ["bG2_f_f", "b1_b1_bG2", "b1_bG2_f"],
]

bispec_bias_sorted_kernels = [
    "b1_b1_b1",
    "b1_b1_b2",
    "b1_b1_bG2",
    "b1_b1_f",
    "b1_b1_b1_f",
    "b1_b1_f_f",
    "b1_b2_f",
    "b1_bG2_f",
    "b1_f_f",
    "b1_f_f_f",
    "b2_f_f",
    "bG2_f_f",
    "f_f_f",
    "f_f_f_f",
    "c1_b1_b1",
    "c1_b1_b2",
    "c1_b1_bG2",
    "c1_b1_f",
    "c1_b1_b1_f",
    "c1_b1_f_f",
    "c1_b2_f",
    "c1_bG2_f",
    "c1_f_f",
    "c1_f_f_f",
    "c1_c1_b1",
    "c1_c1_b2",
    "c1_c1_bG2",
    "c1_c1_f",
    "c1_c1_b1_f",
    "c1_c1_f_f",
    "c2_b1_b1",
    "c2_b1_b2",
    "c2_b1_bG2",
    "c2_b1_f",
    "c2_b1_b1_f",
    "c2_b1_f_f",
    "c2_b2_f",
    "c2_bG2_f",
    "c2_f_f",
    "c2_f_f_f",
    "c2_c1_b1",
    "c2_c1_b2",
    "c2_c1_bG2",
    "c2_c1_f",
    "c2_c1_b1_f",
    "c2_c1_f_f",
    "c2_c2_b1",
    "c2_c2_b2",
    "c2_c2_bG2",
    "c2_c2_f",
    "c2_c2_b1_f",
    "c2_c2_f_f",
    "Bshot_b1_b1",
    "Bshot_b1_f",
    "Bshot_b1_c1",
    "Bshot_b1_c2",
    "Pshot_f_b1",
    "Pshot_f_f",
    "Pshot_f_c1",
    "Pshot_f_c2",
    "fnlloc_b1_b1_b1",
    "fnlloc_b1_b1_f",
    "fnlloc_b1_f_f",
    "fnlloc_f_f_f",
    "fnlequi_b1_b1_b1",
    "fnlequi_b1_b1_f",
    "fnlequi_b1_f_f",
    "fnlequi_f_f_f",
    "fnlortho_b1_b1_b1",
    "fnlortho_b1_b1_f",
    "fnlortho_b1_f_f",
    "fnlortho_f_f_f",
    "fnlortho_LSS_b1_b1_b1",
    "fnlortho_LSS_b1_b1_f",
    "fnlortho_LSS_b1_f_f",
    "fnlortho_LSS_f_f_f",
]


# The first 4 terms in this list are bispectrum kernels,
# the last 4 are power spectrum kernels.
shot_list = ["b1_b1", "b1_c1", "b1_c2", "b1_f", "f_b1", "f_c1", "f_c2", "f_f"]


def group_info(group, file_list=False):
    """
    Args:
        group (int, str) : Group identifier. ``int`` in range 0-6,
         or ``shot``.
        file_list (bool) : If ``True`` returns list of file containing
         the group kernels. Default is ``False``.

    Returns:
        Information about the kernel group.
    """

    if file_list:
        if type(group) is str:
            return shot_list
        else:
            return bispec_bias_lists[group]
    else:
        if type(group) is str:
            return len(shot_list)
        else:
            return len(bispec_bias_lists[group])


def combine_kernels(
    kernel_groups, b1=None, b2=None, bG2=None, c1=None, c2=None, groups=None
):
    """
    Function for combing bias parameters and kernels.

    Args:
        kernel_groups (list) : List of arrays containing kernels.
         Each list element should have shape ``(n_ker, n_samp, n_k)``.
         With ``n_samp`` being the number of predictions to be made,
         ``n_ker`` being the number of kernels in that group,
         and ``n_k`` being the number of k-values.
         Assumes growth rate f is inlcuded in the kernel.
        f (array) : Array containing growth rate.
         Should habe shape ``(n_samp,)``.
        b1 (array) : Array containig b1 values.
         Should habe shape ``(n_samp,)``. Default is ``None``.
        b2 (array) : Array containig b2 values.
         Should habe shape ``(n_samp,)``. Default is ``None``.
        bG2 (array) : Array containig bG2 values.
         Should habe shape ``(n_samp,)``. Default is ``None``.
        c1 (array) : Array containig c1 values.
         Should habe shape ``(n_samp,)``. Default is ``None``.
        c2 (array) : Array containig c2 values.
         Should habe shape ``(n_samp,)``. Default is ``None``.
        groups (list) : List of ``int`` that correspond to the group ids
         for the groups in ``kernel_groups``. Default is ``None``,
         in which case it is assumed all groups are passed.

    Returns:
        Array containig the combined kernels and bias parameters.
        Each list item will have shape ``(n_samp, n_k)``.
    """

    # Determine the number of predictions being made.
    # This is a bit hacky.
    # TODO: Find a nice way of determining the number of predictions.
    npred = kernel_groups[0].shape[1]

    # Store passed bias parameters in dict.
    bias_dict = {"b1": b1, "b2": b2, "bG2": bG2, "c1": c1, "c2": c2}

    # If passsed groups is None. Assume all.
    if groups is None:
        groups = np.arange(len(bispec_bias_lists))

    result = 0

    # Loop over the groups
    for list_id, group in enumerate(groups):

        # Determine what bias parameters are required for each kernel and
        # store in a list such that the bias_dict can be indexed.
        bias_list = [i.split("_") for i in group_info(group, file_list=True)]

        for component_id, components in enumerate(bias_list):

            bi = np.ones((npred,))
            for bij in components:
                if bij != 'f':
                    bi *= bias_dict[bij]

            result += bi.reshape(-1, 1) * kernel_groups[list_id][component_id]

    return result


def mono_bias(bias):
    """
    Function for combining bias parameters for power spectrum monopole.

    Args:
        bias (array) : Array of bias parameters ``{b1, b2, bG2, bGamm3, csl}``.

    Returns:
        Array of shape (n, 16). If all bias parameters were floats ``n=1``.
    """

    bias = np.atleast_2d(bias)

    return np.vstack(
        [
            np.repeat(1.0, bias.shape[0]),
            np.repeat(1.0, bias.shape[0]),
            bias[:, 0],
            bias[:, 0],
            bias[:, 0] ** 2,
            bias[:, 0] ** 2,
            0.25 * bias[:, 1] ** 2,
            bias[:, 0] * bias[:, 1],
            bias[:, 1],
            bias[:, 0] * bias[:, 2],
            bias[:, 2],
            bias[:, 1] * bias[:, 2],
            bias[:, 2] ** 2,
            2 * bias[:, 5],
            (2 * bias[:, 2] + 0.8 * bias[:, 3]) * bias[:, 0],
            2 * bias[:, 2] + 0.8 * bias[:, 3],
        ]
    ).T


def quad_bias(bias):
    """
    Function for combining bias parameters for power spectrum quadrupole.

    Args:
        bias (array) : Array of bias parameters ``{b1, b2, bG2, bGamm3, b4, csl}``.

    Returns:
        Array of shape (n, 11). If all bias parameters were floats ``n=1``.
    """

    bias = np.atleast_2d(bias)

    return np.vstack(
        [
            np.repeat(1.0, bias.shape[0]),
            np.repeat(1.0, bias.shape[0]),
            bias[:, 0],
            bias[:, 0],
            bias[:, 0] ** 2,
            bias[:, 0] * bias[:, 1],
            bias[:, 1],
            bias[:, 0] * bias[:, 2],
            bias[:, 2],
            2 * bias[:, 5],
            2 * bias[:, 2] + 0.8 * bias[:, 3],
        ]
    ).T


def hex_bias(bias):
    """
    Function for combining bias parameters for power spectrum quadrupole.

    Args:
        bias (array) : Array of bias parameters ``{b1, b2, bG2, bGamm3, b4, csl}``.

    Returns:
        Array of shape (n, 11). If all bias parameters were floats ``n=1``.
    """

    bias = np.atleast_2d(bias)

    return np.vstack(
        [
            np.repeat(1.0, bias.shape[0]),
            np.repeat(1.0, bias.shape[0]),
            bias[:, 0],
            bias[:, 0] ** 2,
            bias[:, 1],
            bias[:, 2],
            2 * bias[:, 5],
        ]
    ).T


def extra_bias(bias, multipole):
    """
    Function for combining bias parameters for the terms that are common
    for all multipoles.

    Args:
        bias (array) : Array of bias parameters ``{b1, b2, bG2, bGamm3, b4, csl, cst}``.

        multipole (int) : Multipole order. Must corespond to ``P_n``. Can be
         ``0``, ``2``, or ``4``.

    Returns:
        Array of shape (n, 3). If all bias parameters were floats ``n=1``.
    """

    bias = np.atleast_2d(bias)

    constants_dict = {
        0: [7 / 8, 5 / 4, 35 / 72],
        2: [55 / 22, 275 / 66, 175 / 99],
        4: [1, 390 / 143, 210 / 143],
    }
    constants = constants_dict.get(multipole)

    return np.vstack(
        [
            bias[:, 4] * constants[0] * bias[:, 0] ** 2,
            bias[:, 4] * constants[1] * bias[:, 0],
            bias[:, 4] * constants[2],
        ]
    ).T


comb_bias_fdict = {0: mono_bias, 2: quad_bias, 4: hex_bias, "extra": extra_bias}


def powerspec_multipole(P_n, bias, multipole):
    """
    Function for computing power spectrum multipoles via a
    combination of bias parameters and kernels.

    Args:
        P_n (tuple) : Tuple of arrays containing power spectrum kernels.
         The first element should be those kernels that are unique to
         ``multipole``, the second should be those that are shared by all
         multipoles.
        bias (array) : Array of bias parameters ``{b1, b2, bG2, bGamm3, b4, csl, cst}``.
        multipole (int) : Multipole order. Must corespond to ``P_n``. Can be
         ``0``, ``2``, or ``4``.

    Returns:
        Array containing power spectrum multipole predictions.
    """

    combo_bs = comb_bias_fdict[multipole](bias)
    combo_bs_ext = comb_bias_fdict["extra"](bias, multipole)
    return np.einsum("nb,nbx->nx", combo_bs, P_n[0]) + np.einsum(
        "nb,nbx->nx", combo_bs_ext, P_n[1]
    )
