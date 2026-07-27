import math
import numpy as np
import networkx as nx
import torch
from PIL import Image
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.text import Text
from colour import Color
import shapiq

from .utils import get_subset, sort_interactions, get_max_min_values, get_conditioned_interactions


NORMAL_NODE_SIZE = 0.125  # 0.125
BASE_ALPHA_VALUE = 1.0  # the transparency level for the highest interaction
BASE_SIZE = 0.05  # the size of the highest interaction edge (with scale factor 1)
ADJUST_NODE_ALPHA = True

RED = Color("#ff0d57")
BLUE = Color("#1e88e5")
NEUTRAL = Color("#ffffff")
LINES = Color("#cccccc")

COLORS_K_SII = [
    "#D81B60",
    "#FFB000",
    "#1E88E5",
    "#FE6100",
    "#7F975F",
    "#74ced2",
    "#708090",
    "#9966CC",
    "#CCCCCC",
    "#800080",
]
COLORS_K_SII = COLORS_K_SII * (100 + (len(COLORS_K_SII)))  # repeat the colors list


def draw_fancy_hyper_edges(
    axis: plt.axis,
    pos: dict,
    colors: dict,
    hyper_edges: list[tuple],
    fig,
    debug=False
) -> None:
    """Draws a collection of hyper-edges as a fancy hyper-edge on the graph.

    Note:
        This is also used to draw normal 2-way edges in a fancy way.

    Args:
        axis: The axis to draw the hyper-edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the hyper-edges on.
        hyper_edges: The hyper-edges to draw.
    """
    for hyper_edge in hyper_edges:

        # store all paths for the hyper-edge to combine them later
        all_paths = []

        # make also normal (2-way) edges plottable -> one node becomes the "center" node
        is_hyper_edge = True
        if len(hyper_edge) == 2:
            u, v = hyper_edge
            center_pos = pos[v]
            is_hyper_edge = False
        else:  # a hyper-edge encodes its information in an artificial "center" node
            raise NotImplementedError(f"Only order 2 is supported, but got {len(hyper_edge)}")
            center_pos = pos[hyper_edge]

        color = colors[hyper_edge][:3]
        alpha = colors[hyper_edge][3]
        node_size = 0.1  * alpha
        if debug:
            print("hyper_edge:", hyper_edge, "color", color, "alpha", alpha, "node_size", node_size)

        alpha = min(1.0, max(0.0, alpha))

        # draw the connection point of the hyper-edge
        circle = mpath.Path.circle(center_pos, radius=node_size / 2)
        all_paths.append(circle)
        axis.scatter(center_pos[0], center_pos[1], s=0, c="none", lw=0)  # add empty point for limit

        # draw the fancy connections from the other nodes to the center node
        for player in hyper_edge:

            player_pos = pos[player]

            circle_p = mpath.Path.circle(player_pos, radius=node_size / 2)
            all_paths.append(circle_p)
            axis.scatter(player_pos[0], player_pos[1], s=0, c="none", lw=0)  # for axis limits

            # get the direction of the connection
            direction = (center_pos[0] - player_pos[0], center_pos[1] - player_pos[1])
            direction = np.array(direction) / np.linalg.norm(direction)

            # get 90 degree of the direction
            direction_90 = np.array([-direction[1], direction[0]])

            # get the distance between the player and the center node
            distance = np.linalg.norm(center_pos - player_pos)

            # get the position of the start and end of the connection
            start_pos = player_pos - direction_90 * (node_size / 2)
            middle_pos = player_pos + direction * distance / 2
            end_pos_one = center_pos - direction_90 * (node_size / 2)
            end_pos_two = center_pos + direction_90 * (node_size / 2)
            start_pos_two = player_pos + direction_90 * (node_size / 2)

            # create the connection
            connection = mpath.Path(
                [
                    start_pos,
                    middle_pos,
                    end_pos_one,
                    end_pos_two,
                    middle_pos,
                    start_pos_two,
                    start_pos,
                ],
                [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                ],
            )

            # add the connection to the list of all paths
            all_paths.append(connection)

            # break after the first hyper-edge if there are only two players
            if not is_hyper_edge:
                break

        # combine all paths into one patch
        combined_path = mpath.Path.make_compound_path(*all_paths)
        patch = mpatches.PathPatch(combined_path, facecolor=color, lw=0, alpha=alpha, zorder=5)
        # patch.set_transform(fig.transFigure)

        axis.add_patch(patch)


def draw_normal_hyper_edges(
    axis: plt.axis,
    pos: dict,
    colors: dict,
    hyper_edges: list[tuple],
    fig,
    debug=False
) -> None:
    """Draws a collection of hyper-edges as a fancy hyper-edge on the graph.

    Note:
        This is also used to draw normal 2-way edges in a fancy way.

    Args:
        axis: The axis to draw the hyper-edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the hyper-edges on.
        hyper_edges: The hyper-edges to draw.
    """
    for hyper_edge in hyper_edges:

        # store all paths for the hyper-edge to combine them later
        all_paths = []


        color = colors[hyper_edge][:3]
        alpha = colors[hyper_edge][3]
        node_size = 0.1  * alpha
        if debug:
            print("hyper_edge:", hyper_edge, "color", color, "alpha", alpha, "node_size", node_size)

        alpha = min(1.0, max(0.0, alpha))

        # draw the fancy connections from the other nodes to the center node
        positions = []
        for player in hyper_edge:
            positions.append(pos[player])

        positions = np.array(positions)
        axis.plot(positions[:, 0], positions[:, 1], c=color, alpha=alpha)


def plot_sentence(
    iv: shapiq.InteractionValues,
    sentence: list[str],
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:

    sentence_plot = iv.plot_sentence(
        words=sentence,
        show=False,
        **kwargs,
    )

    return sentence_plot


def value_to_color(value: float) -> tuple[float, float, float]:
    """Converts a negative value to blue and a positive value to red."""
    color = shapiq.plot._config.RED
    if value < 0:
        color = shapiq.plot._config.BLUE
    return float(color.get_red()), float(color.get_green()), float(color.get_blue())


def own_resize(img: np.ndarray, multiples: int = 12) -> np.ndarray:
    """Resizes the image to the given size by repeating the pixels in both directions:
    """
    height, width, n_channels = img.shape
    assert height == width, "The image must be square."
    new_size = height * multiples
    new_img = np.zeros((new_size, new_size, n_channels))
    for i in range(new_size):
        for j in range(new_size):
            new_img[i, j] = img[i // multiples, j // multiples]
    return new_img


def denormalize(img: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """Denormalizes the image given the mean and standard deviation."""
    return img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)


def image_torch_to_array(
    image: torch.Tensor,
    image_mean: torch.Tensor,
    image_std: torch.Tensor
) -> np.ndarray:
    """Converts a torch tensor containing an image to a numpy array and denormalizes it."""
    if len(image.shape) == 4:
        image = image.squeeze(0)
    image = (denormalize(image, image_mean, image_std))
    image = image.permute(1, 2, 0).numpy()
    return image


def interactions_to_color(
    iv: shapiq.InteractionValues,
    max_value: float | None = None
) -> dict[tuple[int, ...], tuple[float, float, float, float]]:
    """Gets the color and alpha value for each interaction in the InteractionValues object.

    Args:
        iv: The InteractionValues object.
        max_value: A maximum value to scale the colors and alpha values with. Defaults to None.
            This is helpful to compare the interactions across different plots.

    Returns:
        A dictionary with the interaction as the key and the color and alpha value as the value.
    """
    if max_value is None:
        max_value, _ = get_max_min_values([iv])
    colors = {}
    for interaction in iv.interaction_lookup.keys():
        if len(interaction) == 0:
            continue
        score = iv[interaction]
        color = value_to_color(score)
        if max_value is None:
            alpha = 0.75
        else:
            alpha = abs(score) / max_value
        alpha = float(np.min([alpha, 1]))
        colors[interaction] = (*color, alpha)
    return colors


def interactions_to_heatmap(
    iv: shapiq.InteractionValues,
    img: Image.Image | np.ndarray,
    indices: list[tuple[int, ...]],
    colors: dict[tuple[int, ...], tuple[float, float, float, float]] | None = None,
) -> Image.Image:
    """Turns the InteractionValues into a heatmap."""
    # get image size information
    image_size = img.shape[0]
    grid_size = int(np.sqrt(iv.n_players))
    patch_size = int(image_size / grid_size)
    # get the values

    max_abs_value = float(np.quantile(np.abs(iv.get_n_order(order=1).values), 0.99))
    # create the heatmap array
    heatmap = np.zeros((grid_size, grid_size, 4))
    for i, c_index in enumerate(indices):
        row = i // grid_size
        column = i % grid_size
        if colors is not None:
            # use the color from the dictionary
            color_patch = colors[(c_index,)]
            alpha = color_patch[3]
            color = (*color_patch[:3], np.min([alpha, 1]))
        else:
            score_patch = iv[(c_index,)]
            color_patch = value_to_color(score_patch)
            alpha = abs(score_patch) / max_abs_value
            color = (*color_patch, np.min([alpha, 1]))
        heatmap[row, column] = color
    heatmap = own_resize(heatmap, multiples=patch_size)
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    return heatmap


def plot_heatmap(sv: shapiq.InteractionValues, img: Image.Image, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """Plots the Shapley values as a heatmap onto the image.
    """
    image_size = img.shape[0]
    grid_size = int(np.sqrt(sv.n_players))
    patch_size = int(image_size / grid_size)

    sv_values_without_baseline = np.array([sv.dict_values[(i,)] for i in range(sv.n_players)])
    max_abs_value = float(np.quantile(np.abs(sv_values_without_baseline), 0.99))

    sv_image = np.zeros((grid_size, grid_size, 4))
    for i in range(sv.n_players):
        row = i // grid_size
        column = i % grid_size
        sv_patch = sv_values_without_baseline[i]

        color_patch = value_to_color(sv_patch)
        alpha = abs(sv_patch) / max_abs_value
        color = (*color_patch, np.min([alpha, 1]))
        sv_image[row, column] = color

    sv_image = own_resize(sv_image, multiples=patch_size)
    # sv_image = interactions_to_heatmap(sv, img)

    fig, ax = plt.subplots(**kwargs)
    ax.imshow(img, alpha=0.9)
    ax.imshow(sv_image)
    ax.axis("off")
    plt.tight_layout()
    return fig, ax


def _get_word_dimensions(words: list[str], figsize: tuple[int, int], font_size: int) -> list[tuple[float, float]]:
    """Creates an axis object, draws the words on it and returns the dimensions of the words for
    later use."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    font = FontProperties(family='sans-serif', style='normal', size=font_size)
    word_dimensions = []
    for word in words:
        text = Text(
            x=0,
            y=0,
            text=word,
            fontproperties=font
        )
        ax.add_artist(text)
        width = text.get_window_extent().width
        height = text.get_window_extent().height
        word_dimensions.append((width, height))
    plt.close()
    return word_dimensions


def _get_line_lengths(
    word_dimensions: list[tuple[float, float]],
    word_spacing: float,
    line_start: float,
    line_end: float,
) -> list[float]:
    """Calculates the lengths of the lines in the text."""
    line_lengths = []
    x_pos = line_start
    added_line = False
    for i, dim in enumerate(word_dimensions):
        width, height = dim
        if x_pos + width > line_end:
            line_lengths.append(float(x_pos - line_start))
            x_pos = line_start
            added_line = True
        else:
            added_line = False
        x_pos += width + word_spacing
    if not added_line:
        line_lengths.append(float(x_pos - line_start))
    return line_lengths


def plot_image_and_text_together(
    img: Image.Image | np.ndarray | dict,
    text: list[str],
    image_players: list[int] | dict[str, int] | dict[str, list[int]],
    iv: shapiq.InteractionValues,
    *,
    player_mask: set[int | tuple[int, ...]] | None = None,
    color_mask_white: bool = False,
    opacity_white: float = 0.8,
    plot_heatmap: bool = True,
    sort_by_abs: bool = True,
    fontsize: int = 30,
    top_k: int = 50,
    max_value: float | None = None,
    debug = False,
    figsize = (9, 10),
    image_span: float = 0.9,
    margin: float = 0,
    line_padding: float = 0.5,
    margin_text: tuple[float, float] = (0., 0.),
    color_text: bool = True,
    plot_interactions: bool = False,
    normalize_jointly: bool = True,
    condition_on_player: int | None = None,
    show=True,
    ax=None,
    axes: list[plt.Axes] | None = None,  # ADDED: list of axes for modalities
) -> tuple[plt.Figure, plt.Axes, dict[int, tuple[float, float]]]:
    """Extended: same behaviour as original but supports multiple image modalities.

    img can be:
      - a single square numpy array (old behaviour),
      - a dict mapping modality name -> numpy array.

    image_players can be:
      - old list[int] (old behaviour),
      - dict[str, int] mapping modality -> number_of_patches (players are assigned sequentially
        starting from 0),
      - dict[str, list[int]] mapping modality -> explicit player indices.

    Notes / behaviour:
      - When image_players is a dict of counts we assume image player indices start at 0 and are
        contiguous; text players are the remaining player indices in sorted order.
      - When image_players is a dict of lists we use those indices directly.
      - Each modality's image is arranged in its own square grid and the modalities are laid out
        left-to-right inside the image area in proportion to their number of patches.
    """

    if condition_on_player is not None:
        iv = get_conditioned_interactions(iv, player=condition_on_player, divide=False)

    # Normalize image_players to modality -> list[int]
    if isinstance(image_players, dict):
        modality_to_indices = {}
        if all(isinstance(v, int) for v in image_players.values()):
            # counts: assign sequential indices starting at 0
            idx = 0
            for mod, count in image_players.items():
                modality_to_indices[mod] = list(range(idx, idx + count))
                idx += count
        else:
            # assume lists
            for mod, v in image_players.items():
                modality_to_indices[mod] = list(v)
    else:
        # legacy single-modality list
        modality_to_indices = {"_default": list(image_players)}

    # Normalize img to modality->array
    if isinstance(img, dict):
        images = img
    else:
        # single image provided -> reuse for all modalities (legacy)
        first_mod = list(modality_to_indices.keys())[0]
        images = {mod: img for mod in modality_to_indices.keys()}

    # Collect global image player indices and counts
    all_image_indices = []
    for lst in modality_to_indices.values():
        all_image_indices.extend(lst)
    total_image_players = len(all_image_indices)
    if total_image_players == 0:
        raise ValueError("No image players provided.")

    # Colors (joint or per-modality as fallback)
    if normalize_jointly:
        colors = interactions_to_color(iv, max_value=max_value)
    else:
        colors = {}
        # per modality
        for mod, indices in modality_to_indices.items():
            iv_mod = get_subset(iv, players=indices)
            colors_mod = interactions_to_color(iv_mod, max_value=max_value)
            colors.update(colors_mod)
        # text
        text_players = sorted(set(range(iv.n_players)) - set(all_image_indices))
        iv_text = get_subset(iv, players=text_players, rename_players=False)
        colors_text = interactions_to_color(iv_text, max_value=max_value)
        colors.update(colors_text)

    # Prepare figure / axis
    if axes is not None:
        # use provided axes for each modality; create a full-figure overlay axis for text/positions
        fig = axes[0].figure
        if not ax:
            ax = fig.add_axes([0, 0, 1, 1], zorder=10, aspect=0.5)
            ax.set_position(axes[0].get_position())
            # ax = fig.add_axes([0, 0, 1, 1], zorder=10, position=[0, 0, 1, 1])
            # ax = axes[0].inset_axes([0, 0, 1, 1])
            # print(ax.get_position())
            # ax.set_aspect('equal', adjustable='box')
        else:
            fig = ax.figure
        # ensure modality axes are clean
        for a in axes:
            a.axis("off")
    else:
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
    # ax.axis('off')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)

    # Compute overall image extent (centered as before)
    image_aspect_ratio = 1  # images are assumed square
    fig_aspect_ratio = figsize[0] / figsize[1]
    image_height = image_span
    image_width = image_height * image_aspect_ratio / fig_aspect_ratio
    x_center_image = 0.5
    left = x_center_image - image_width / 2
    right = x_center_image + image_width / 2
    bottom = 1 - image_span
    top = 1
    extent = (left, right, bottom, top)
    # extent = (0, 0, 1, 1)

    # draw a transparent background image to get window extents (used for text baseline)
    sample_mod = next(iter(images))
    sample_img = images[sample_mod]
    # image_patch = ax.imshow(sample_img, extent=extent, zorder=0, alpha=0.0)
    # image_patch = ax.imshow(sample_img, extent=extent, aspect=image_aspect_ratio, zorder=0, alpha=0.0)

    # For each modality: compute relative width proportional to its number of patches and draw image + heatmap
    left_current = left
    positions: dict[int, tuple[float, float]] = {}
    for mod_idx, (mod, indices) in enumerate(modality_to_indices.items()):
        n_mod = len(indices)
        # get iv subset for this modality (used by heatmap/masking)
        iv_mod = get_subset(iv, players=indices)

        if axes is not None:
            # use the provided axis bbox for this modality (figure coords)
            bbox = axes[mod_idx].get_position()
            mod_left = bbox.x0
            mod_right = bbox.x0 + bbox.width
            mod_width = bbox.width
            bottom_mod = bbox.y0
            top_mod = bbox.y0 + bbox.height
            print(bbox)
            image_patches = image_into_patches(images.get(mod, sample_img), n_mod)
        else:
            width_prop = n_mod / total_image_players
            mod_width = image_width * width_prop
            mod_left = left_current
            mod_right = mod_left + mod_width
            left_current = mod_right

            img_mod = images.get(mod, sample_img)
            # Create patches for modality image and overlay
            image_patches = image_into_patches(img_mod, n_mod)
            # grid_size for patch placement (legacy)
            bbox = None
            bottom_mod = bottom
            top_mod = top

        img_mod = images.get(mod, sample_img)

        # draw modality image (either into its own axis or into the combined axis with extent)
        if axes is not None:
            img_ax = axes[mod_idx]
            img_ax.imshow(img_mod, aspect=image_aspect_ratio, zorder=0)
        else:
            ax.imshow(img_mod, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect='auto', zorder=0)

        # heatmap for modality
        if plot_heatmap:
            heatmap = interactions_to_heatmap(iv=iv_mod, img=img_mod, colors=colors, indices=indices)
            if axes is not None:
                axes[mod_idx].imshow(heatmap, aspect=image_aspect_ratio, zorder=1)
            else:
                ax.imshow(heatmap, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect='auto', zorder=1)

        # mask players not in player_mask for this modality
        if player_mask is not None:
            bw_colors = {}
            for player in iv_mod.interaction_lookup.keys():
                if len(player) > 0 and (player in player_mask or player[0] in player_mask):
                    bw_colors[player] = (0.0, 0.0, 0.0, 0)
                else:
                    if color_mask_white:
                        bw_colors[player] = (1, 1, 1, opacity_white)
                    else:
                        bw_colors[player] = (0.5, 0.5, 0.5, 1)
            bw_heatmap = interactions_to_heatmap(iv=iv_mod, img=img_mod, colors=bw_colors)
            if axes is not None:
                axes[mod_idx].imshow(bw_heatmap, aspect=image_aspect_ratio, zorder=2)
            else:
                ax.imshow(bw_heatmap, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect='auto', zorder=2)

        # compute positions for each patch in this modality
        grid_size = int(np.sqrt(n_mod))
        for i, player_global in enumerate(indices):
            col = i % grid_size
            row = i // grid_size
            # compute using the modality bounds (works for both provided axes (figure coords) and combined axis coords)
            x = mod_left + (mod_width / grid_size) * col #(col + 0.5)
            y = top_mod - ( (top_mod - bottom_mod) / grid_size ) * row #(row + 0.5)
            positions[player_global] = (x, y)
            if condition_on_player is not None and condition_on_player == player_global:
                left_rectangle = x - (mod_width / grid_size) / 2
                bottom_rectangle = y - ((top_mod - bottom_mod) / grid_size) / 2
                width_rectangle = mod_width / grid_size
                height_rectangle = (top_mod - bottom_mod) / grid_size
                outline_rectangle = mpatches.Rectangle(
                    (left_rectangle, bottom_rectangle),
                    width_rectangle,
                    height_rectangle,
                    linewidth=6,
                    edgecolor='black',
                    facecolor='none',
                    linestyle='--',
                )
                ax.add_patch(outline_rectangle)

    # compute conversion and text baseline using drawn (invisible) image_patch
    # x0, y0, width_px, height_px = image_patch.get_window_extent().bounds
    # conversion_factor = figsize[0] * 100
    # space_width = _get_word_dimensions(words=["____"], figsize=figsize, font_size=fontsize)[0][0]
    # line_height = _get_word_dimensions(words=["A"], figsize=figsize, font_size=fontsize)[0][1]

    # # compute line starts/ends in same way as before
    # x0_conv = 0
    # x1_conv = figsize[0] * 100
    # word_dimensions = _get_word_dimensions(words=text, figsize=figsize, font_size=fontsize)
    # word_spacing = space_width * 1.5
    # line_start = x0_conv + space_width * 2 + margin_text[0] * (x1_conv - x0_conv)
    # line_end = x1_conv - space_width * 2 - margin_text[1] * (x1_conv - x0_conv)
    # line_lengths = _get_line_lengths(
    #     word_dimensions=word_dimensions,
    #     word_spacing=word_spacing,
    #     line_start=line_start,
    #     line_end=line_end,
    # )
    # left_margins = [(line_end - line_start - ll) / 2 for ll in line_lengths]

    # # determine text player indices (remaining players)
    # text_player_indices = sorted(set(range(iv.n_players)) - set(all_image_indices))
    # if len(text_player_indices) < len(text):
    #     # fallback to legacy contiguous assignment if counts don't match
    #     start = total_image_players
    #     text_player_indices = list(range(start, start + len(text)))

    # # render text: place text words and register positions keyed by global player index
    # line_counter = 0
    # x_pos = line_start + left_margins[line_counter]
    # y_pos = y0 - line_height * (1.0 + margin)
    # for i, (word, dim) in enumerate(zip(text, word_dimensions)):
        # width_w, height_w = dim
        # player_global = text_player_indices[i]
        # if x_pos + width_w > line_end:
        #     line_counter += 1
        #     try:
        #         x_pos = line_start + left_margins[line_counter]
        #     except IndexError:
        #         x_pos = line_start + left_margins[-1]
        #     y_pos -= line_height + line_height * line_padding

        # color = colors.get((player_global,), (1,1,1,1))[:3]
        # alpha = colors.get((player_global,), (1,1,1,1))[3]
        # if not color_text:
        #     color = "white"
        #     text_color = "black"
        # else:
        #     text_color = "black" if alpha < 0.8 else "white"

        # bbox = {
        #     "facecolor": color,
        #     "alpha": alpha,
        #     "edgecolor": color,
        #     "boxstyle": "round,pad=0.1",
        #     "linestyle": "solid",
        #     "linewidth": 1.5
        # }
        # if not color_text:
        #     bbox["edgecolor"] = "lightgrey"
        #     bbox["alpha"] = 1

        # if player_mask is not None:
        #     bbox["alpha"] = 1
        #     if (player_global,) in player_mask or player_global in player_mask:
        #         bbox["edgecolor"] = "lightgrey"
        #     else:
        #         if color_mask_white:
        #             bbox["facecolor"] = "white"
        #             bbox["edgecolor"] = "white"
        #             text_color = "white"
        #         else:
        #             bbox["facecolor"] = "lightgrey"
        #             bbox["edgecolor"] = "lightgrey"
        #             text_color = "lightgrey"

        # if condition_on_player is not None and player_global == condition_on_player:
        #     bbox["edgecolor"] = "black"
        #     bbox["linewidth"] = 6
        #     bbox["alpha"] = 1
        #     bbox["facecolor"] = "white"
        #     text_color = "black"
        #     bbox["linestyle"]= "--"

        # ax.text(
        #     x_pos / conversion_factor,
        #     y_pos / conversion_factor,
        #     word,
        #     fontsize=fontsize,
        #     bbox=bbox,
        #     zorder=100,
        #     color=text_color,
        # )
        # x_pos_center = (x_pos + width_w / 2) / conversion_factor
        # y_pos_center = (y_pos + line_height / 2) / conversion_factor
        # positions[player_global] = (x_pos_center, y_pos_center)
        # x_pos += width_w + word_spacing

    # add interactions (uses positions keyed by global player indices)
    if iv.max_order >= 2 and plot_interactions:
        iv_second_order = iv.get_n_order(order=2, min_order=2, max_order=2)
        top_interactions = sort_interactions(iv_second_order, reverse=True, sort_by_abs=sort_by_abs)[:top_k]

        hyper_edges_to_draw = []
        positions_to_draw = {}
        for inter in top_interactions:
            interaction = inter[0]
            hyper_edges_to_draw.append(interaction)
            for player in interaction:
                if player in positions:
                    positions_to_draw[player] = np.array([positions[player][0], positions[player][1]])
                    # positions_to_draw[player] = np.array([0, 0])

        draw_fancy_hyper_edges(
            axis=ax,
            pos=positions_to_draw,
            colors=colors,
            hyper_edges=hyper_edges_to_draw,
        )

    plt.tight_layout(pad=0.05)
    if not show:
        return fig, ax, positions


def plot_image_and_image_together(
    img: Image.Image | np.ndarray | dict,
    image_players: list[int] | dict[str, int] | dict[str, list[int]],
    iv: shapiq.InteractionValues,
    *,
    player_mask: set[int | tuple[int, ...]] | None = None,
    color_mask_white: bool = False,
    opacity_white: float = 0.8,
    plot_heatmap: bool = True,
    sort_by_abs: bool = True,
    top_k: int = 50,
    max_value: float | None = None,
    debug = False,
    figsize = (9, 10),
    image_span: float = 0.9,
    plot_interactions: bool = False,
    normalize_jointly: bool = True,
    condition_on_player: int | None = None,
    show=True,
    ax=None,
    axes: list[plt.Axes] | None = None,  # ADDED: list of axes for modalities
) -> tuple[plt.Figure, plt.Axes, dict[int, tuple[float, float]]]:
    """Extended: same behaviour as original but supports multiple image modalities.

    img can be:
      - a single square numpy array (old behaviour),
      - a dict mapping modality name -> numpy array.

    image_players can be:
      - old list[int] (old behaviour),
      - dict[str, int] mapping modality -> number_of_patches (players are assigned sequentially
        starting from 0),
      - dict[str, list[int]] mapping modality -> explicit player indices.

    Notes / behaviour:
      - When image_players is a dict of counts we assume image player indices start at 0 and are
        contiguous; text players are the remaining player indices in sorted order.
      - When image_players is a dict of lists we use those indices directly.
      - Each modality's image is arranged in its own square grid and the modalities are laid out
        left-to-right inside the image area in proportion to their number of patches.
    """

    if condition_on_player is not None:
        iv = get_conditioned_interactions(iv, player=condition_on_player, divide=False)

    # Normalize image_players to modality -> list[int]
    if isinstance(image_players, dict):
        modality_to_indices = {}
        if all(isinstance(v, int) for v in image_players.values()):
            # counts: assign sequential indices starting at 0
            idx = 0
            for mod, count in image_players.items():
                modality_to_indices[mod] = list(range(idx, idx + count))
                idx += count
        else:
            # assume lists
            for mod, v in image_players.items():
                modality_to_indices[mod] = list(v)
    else:
        # legacy single-modality list
        modality_to_indices = {"_default": list(image_players)}

    # Normalize img to modality->array
    if isinstance(img, dict):
        images = img
    else:
        # single image provided -> reuse for all modalities (legacy)
        first_mod = list(modality_to_indices.keys())[0]
        images = {mod: img for mod in modality_to_indices.keys()}

    # Collect global image player indices and counts
    all_image_indices = []
    for lst in modality_to_indices.values():
        all_image_indices.extend(lst)
    total_image_players = len(all_image_indices)
    if total_image_players == 0:
        raise ValueError("No image players provided.")

    # Colors (joint or per-modality as fallback)
    if normalize_jointly:
        colors = interactions_to_color(iv, max_value=max_value)
    else:
        colors = {}
        # per modality
        for mod, indices in modality_to_indices.items():
            iv_mod = get_subset(iv, players=indices)
            colors_mod = interactions_to_color(iv_mod, max_value=max_value)
            colors.update(colors_mod)
        # text
        text_players = sorted(set(range(iv.n_players)) - set(all_image_indices))
        iv_text = get_subset(iv, players=text_players, rename_players=False)
        colors_text = interactions_to_color(iv_text, max_value=max_value)
        colors.update(colors_text)

    # Prepare figure / axis
    if axes is not None:
        # use provided axes for each modality; create a full-figure overlay axis for text/positions
        fig = axes[0].figure
        if not ax:
            ax = fig.add_axes([0, 0, .66, 0.66], frameon=False, zorder=10, aspect_ratio=1)
        else:
            fig = ax.figure
        # ensure modality axes are clean
        for a in axes:
            a.axis("off")
    else:
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
            ax.set_aspect(1)
        else:
            fig = ax.figure

    # Compute overall image extent (centered as before)
    image_aspect_ratio = 1  # images are assumed square
    mod_sep = 1 - image_span
    image_height = image_span + mod_sep
    image_width = image_height * image_aspect_ratio
    x_center_image = 0.5
    left = x_center_image - image_width / 2
    bottom = 0
    top = 1

    # draw a transparent background image to get window extents (used for text baseline)
    sample_mod = next(iter(images))
    sample_img = images[sample_mod]

    # For each modality: compute relative width proportional to its number of patches and draw image + heatmap
    left_current = left
    positions: dict[int, tuple[float, float]] = {}
    for mod_idx, (mod, indices) in enumerate(modality_to_indices.items()):
        n_mod = len(indices)
        # get iv subset for this modality (used by heatmap/masking)
        iv_mod = get_subset(iv, players=indices)

        mod_width = image_width
        mod_left = left_current
        mod_right = mod_left + mod_width
        left_current = mod_right + mod_sep

        img_mod = images.get(mod, sample_img)
        # Create patches for modality image and overlay
        bottom_mod = bottom
        top_mod = top

        img_mod = images.get(mod, sample_img)

        # draw modality image (either into its own axis or into the combined axis with extent)
        ax.imshow(img_mod, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect=1, zorder=0)

        # heatmap for modality
        if plot_heatmap:
            heatmap = interactions_to_heatmap(iv=iv_mod, img=img_mod, colors=colors, indices=indices)
            ax.imshow(heatmap, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect=1, zorder=1)

        # mask players not in player_mask for this modality
        if player_mask is not None:
            bw_colors = {}
            for player in iv_mod.interaction_lookup.keys():
                if len(player) > 0 and (player in player_mask or player[0] in player_mask):
                    bw_colors[player] = (0.0, 0.0, 0.0, 0)
                else:
                    if color_mask_white:
                        bw_colors[player] = (1, 1, 1, opacity_white)
                    else:
                        bw_colors[player] = (0.5, 0.5, 0.5, 1)
            bw_heatmap = interactions_to_heatmap(iv=iv_mod, img=img_mod, colors=bw_colors)
            ax.imshow(bw_heatmap, extent=(mod_left, mod_right, bottom_mod, top_mod), aspect='auto', zorder=2)

        # compute positions for each patch in this modality
        grid_size = int(np.sqrt(n_mod))
        for i, player_global in enumerate(indices):
            col = i % grid_size
            row = i // grid_size
            # compute using the modality bounds (works for both provided axes (figure coords) and combined axis coords)
            x = mod_left + (mod_width / grid_size) * (col + 0.5)
            y = top_mod - ( (top_mod - bottom_mod) / grid_size ) * (row + 0.5)
            positions[player_global] = (x, y)
            if condition_on_player is not None and condition_on_player == player_global:
                left_rectangle = x - (mod_width / grid_size) / 2
                bottom_rectangle = y - ((top_mod - bottom_mod) / grid_size) / 2
                width_rectangle = mod_width / grid_size
                height_rectangle = (top_mod - bottom_mod) / grid_size
                outline_rectangle = mpatches.Rectangle(
                    (left_rectangle, bottom_rectangle),
                    width_rectangle,
                    height_rectangle,
                    linewidth=6,
                    edgecolor='black',
                    facecolor='none',
                    linestyle='--',
                )
                ax.add_patch(outline_rectangle)

    # add interactions (uses positions keyed by global player indices)
    if iv.max_order >= 2 and plot_interactions:
        iv_second_order = iv.get_n_order(order=2, min_order=2, max_order=2)
        top_interactions = sort_interactions(iv_second_order, reverse=True, sort_by_abs=sort_by_abs)[:top_k]

        hyper_edges_to_draw = []
        positions_to_draw = {}
        for inter in top_interactions:
            interaction = inter[0]
            hyper_edges_to_draw.append(interaction)
            for player in interaction:
                if player in positions:
                    positions_to_draw[player] = np.array([positions[player][0], positions[player][1]])

        draw_fancy_hyper_edges(
            axis=ax,
            pos=positions_to_draw,
            colors=colors,
            hyper_edges=hyper_edges_to_draw,
            fig=fig,
        )

    if not show:
        return fig, ax, positions


def image_into_patches(img: np.ndarray, n_patches: int) -> list[np.ndarray]:
    """Turns an image into a list of patches.

    Args:
        img: The image to split into patches.
        n_patches: The number of patches to split the image into.

    Returns:
        A list of image patches where the first patch it the top left corner and the last patch is
            the bottom right corner of the image.
    """
    grid_size = int(np.sqrt(n_patches))
    patch_size = img.shape[0] // grid_size
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
            patches.append(patch)
    return patches


def plot_interaction_subset(
    iv: shapiq.InteractionValues,
    clique: set[int],
    image_players: list[int],
    img: np.ndarray,
    text: list[str],
    fontsize: int = 25,
    image_size: float = 1,
    edges_size: float = 3,
    figsize=(4,4),
    max_value=None,
    plot_main_effect: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Visualizes all second order interactions of a clique (a subset of players) with the SI graph.

    Args:
        iv: Interaction values to visualize.
        clique: The players subset to visualize.
        image_players: The players that are the images (the rest are text).
        img: The image to visualize.
        text: The text to visualize.
        fontsize: The font size of the text.
        image_size: A scaling factor for the image patches. The higher the value, the bigger the
            image patches.
        edges_size: Scale factor for the edges of the graph.
        plot_main_effect: Weather to plot the main effect of the clique or not. Defaults to False.

    Returns:
        A tuple of the figure and axes of the plot.
    """
    n_image_players = len(image_players)
    image_patches = image_into_patches(img, n_image_players)

    colors = interactions_to_color(iv, max_value=max_value)

    # get the interaction values for the clique
    iv_subset = iv.get_n_order(order=2)
    iv_subset = iv_subset.get_subset(players=list(clique))
    graph = [interaction for interaction in iv_subset.interaction_lookup.keys()]
    if len(iv_subset) == 0:
        # there are no interactions in the subset we create a dummy interaction to order the elements
        iv_subset = iv.get_n_order(min_order=1, max_order=2)
        iv_subset = iv_subset.get_subset(players=list(clique))
        graph = [interaction for interaction in shapiq.powerset(clique, min_size=2, max_size=2)]

    fig, axis, positions, _ = si_graph_plot(
        interaction_values=iv_subset,
        graph=graph,
        show=False,
        draw_original=False,
        size_factor=edges_size,
        figsize=figsize
    )

    max_position = 1.1
    min_position = -max_position
    axis.set_xlim(min_position, max_position)
    axis.set_ylim(min_position, max_position)

    # for each position add an image or text
    for player, coords in positions.items():
        color = colors[(player,)]
        if player in image_players:
            patch = image_patches[image_players.index(player)]
            patch = (patch * 255).astype(np.uint8)
            imagebox = OffsetImage(patch, zoom=2 * image_size)
            bbox = {
                "facecolor": color,
                "edgecolor": color,
                "boxstyle": "round,pad=0.2",
                "linestyle": "solid",
                "linewidth": 0.5,
            }
            ab = AnnotationBbox(imagebox, coords, frameon=True, bboxprops=bbox)
            axis.add_artist(ab)
            color_patch = np.zeros(shape=(patch.shape[0], patch.shape[1], 4))
            color_patch[:, :, 0] = color[0]
            color_patch[:, :, 1] = color[1]
            color_patch[:, :, 2] = color[2]
            color_patch[:, :, 3] = color[3]
            color_patch = (color_patch * 255).astype(np.uint8)
            imagebox_color = OffsetImage(color_patch, zoom=2 * image_size)
            ab_color = AnnotationBbox(imagebox_color, coords, frameon=False)
            axis.add_artist(ab_color)

        else:
            # first add a white text and then the colored one
            bbox = {
                "facecolor": "white",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.2",
                "linestyle": "solid",
            }
            axis.text(
                coords[0],
                coords[1],
                text[player - n_image_players],
                fontsize=fontsize,
                color='white',
                va='center',
                ha='center',
                bbox=bbox,
            )

            bbox = {
                "facecolor": color,
                "edgecolor": color,
                "boxstyle": "round,pad=0.2",
                "linestyle": "solid",
            }
            axis.text(
                coords[0],
                coords[1],
                text[player - n_image_players],
                fontsize=fontsize,
                color='black' if color[3] < 0.8 else 'white',
                va='center',
                ha='center',
                bbox=bbox,
            )

    # finalize plot
    plt.tight_layout()
    return fig, axis


#####


def get_color(value: float) -> str:
    """Returns blue color for negative values and red color for positive values.

    Args:
        value (float): The value to determine the color for.

    Returns:
        str: The color as a hex string.
    """
    if value >= 0:
        return RED.hex
    return BLUE.hex


def _normalize_value(
    value: float, max_value: float, base_value: float, cubic_scaling: bool = False
) -> float:
    """Scale a value between 0 and 1 based on the maximum value and a base value.

    Args:
        value: The value to normalize/scale.
        max_value: The maximum value to normalize/scale the value by.
        base_value: The base value to scale the value by. For example, the alpha value for the
            highest interaction (as defined in ``BASE_ALPHA_VALUE``) or the size of the highest
            interaction edge (as defined in ``BASE_SIZE``).
        cubic_scaling: Whether to scale cubically (``True``) or linearly (``False``. default)
            between 0 and 1.

    Returns:
        The normalized/scaled value.
    """
    ratio = abs(value) / abs(max_value)  # ratio is always positive in [0, 1]
    if cubic_scaling:
        ratio = ratio**3
    alpha = ratio * base_value
    return alpha


def _draw_fancy_hyper_edges(
    axis: plt.axis,
    pos: dict,
    graph: nx.Graph,
    hyper_edges: list[tuple],
) -> None:
    """Draws a collection of hyper-edges as a fancy hyper-edge on the graph.

    Note:
        This is also used to draw normal 2-way edges in a fancy way.

    Args:
        axis: The axis to draw the hyper-edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the hyper-edges on.
        hyper_edges: The hyper-edges to draw.
    """
    for hyper_edge in hyper_edges:

        # store all paths for the hyper-edge to combine them later
        all_paths = []

        # make also normal (2-way) edges plottable -> one node becomes the "center" node
        is_hyper_edge = True
        if len(hyper_edge) == 2:
            u, v = hyper_edge
            center_pos = pos[v]
            node_size = graph[u][v]["size"]
            color = graph[u][v]["color"]
            alpha = graph[u][v]["alpha"]
            is_hyper_edge = False
        else:  # a hyper-edge encodes its information in an artificial "center" node
            center_pos = pos[hyper_edge]
            node_size = graph.nodes.get(hyper_edge)["size"]
            color = graph.nodes.get(hyper_edge)["color"]
            alpha = graph.nodes.get(hyper_edge)["alpha"]

        alpha = min(1.0, max(0.0, alpha))

        # draw the connection point of the hyper-edge
        circle = mpath.Path.circle(center_pos, radius=node_size / 2)
        all_paths.append(circle)
        axis.scatter(center_pos[0], center_pos[1], s=0, c="none", lw=0)  # add empty point for limit

        # draw the fancy connections from the other nodes to the center node
        for player in hyper_edge:

            player_pos = pos[player]

            circle_p = mpath.Path.circle(player_pos, radius=node_size / 2)
            all_paths.append(circle_p)
            axis.scatter(player_pos[0], player_pos[1], s=0, c="none", lw=0)  # for axis limits

            # get the direction of the connection
            direction = (center_pos[0] - player_pos[0], center_pos[1] - player_pos[1])
            direction = np.array(direction) / np.linalg.norm(direction)

            # get 90 degree of the direction
            direction_90 = np.array([-direction[1], direction[0]])

            # get the distance between the player and the center node
            distance = np.linalg.norm(center_pos - player_pos)

            # get the position of the start and end of the connection
            start_pos = player_pos - direction_90 * (node_size / 2)
            middle_pos = player_pos + direction * distance / 2
            end_pos_one = center_pos - direction_90 * (node_size / 2)
            end_pos_two = center_pos + direction_90 * (node_size / 2)
            start_pos_two = player_pos + direction_90 * (node_size / 2)

            # create the connection
            connection = mpath.Path(
                [
                    start_pos,
                    middle_pos,
                    end_pos_one,
                    end_pos_two,
                    middle_pos,
                    start_pos_two,
                    start_pos,
                ],
                [
                    mpath.Path.MOVETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                    mpath.Path.CURVE3,
                    mpath.Path.CURVE3,
                    mpath.Path.LINETO,
                ],
            )

            # add the connection to the list of all paths
            all_paths.append(connection)

            # break after the first hyper-edge if there are only two players
            if not is_hyper_edge:
                break

        # combine all paths into one patch
        combined_path = mpath.Path.make_compound_path(*all_paths)
        patch = mpatches.PathPatch(combined_path, facecolor=color, lw=0, alpha=alpha)

        axis.add_patch(patch)


def _draw_graph_nodes(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    nodes: list | None = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
) -> None:
    """Draws the nodes of the graph as circles with a fixed size.

    Args:
        ax: The axis to draw the nodes on.
        pos: The positions of the nodes.
        graph: The graph to draw the nodes on.
        nodes: The nodes to draw. If ``None``, all nodes are drawn. Defaults to ``None``.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
    """
    for node in graph.nodes:
        if nodes is not None and node not in nodes:
            continue

        position = pos[node]
        circle = mpath.Path.circle(position, radius=normal_node_size / 2)
        patch = mpatches.PathPatch(circle, facecolor="white", lw=1, alpha=1, edgecolor="black")
        ax.add_patch(patch)

        # add empty scatter for the axis to adjust the limits later
        ax.scatter(position[0], position[1], s=0, c="none", lw=0)


def _draw_explanation_nodes(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    nodes: list | None = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
    node_area_scaling: bool = False,
) -> None:
    """Adds the node level explanations to the graph as circles with varying sizes.

    Args:
        ax: The axis to draw the nodes on.
        pos: The positions of the nodes.
        graph: The graph to draw the nodes on.
        nodes: The nodes to draw. If ``None``, all nodes are drawn. Defaults to ``None``.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
        node_area_scaling: Whether to scale the node sizes based on the area of the nodes (``True``)
            or the radius of the nodes (``False``). Defaults to ``False``.
    """
    for node in graph.nodes:
        if isinstance(node, tuple):
            continue
        if nodes is not None and node not in nodes:
            continue
        position = pos[node]
        color = graph.nodes.get(node)["color"]
        explanation_size = graph.nodes.get(node)["size"]
        alpha = 1.0
        if ADJUST_NODE_ALPHA:
            alpha = graph.nodes.get(node)["alpha"]

        alpha = min(1.0, max(0.0, alpha))

        radius = normal_node_size / 2 + explanation_size / 2
        if node_area_scaling:
            # get the radius of a circle with the same area as the combined area
            normal_node_area = math.pi * (normal_node_size / 2) ** 2
            this_node_area = math.pi * (explanation_size / 2) ** 2
            combined_area = normal_node_area + this_node_area
            radius = math.sqrt(combined_area / math.pi)

        circle = mpath.Path.circle(position, radius=radius)
        patch = mpatches.PathPatch(circle, facecolor=color, lw=1, edgecolor="white", alpha=alpha)
        ax.add_patch(patch)

        ax.scatter(position[0], position[1], s=0, c="none", lw=0)  # add empty point for limits


def _draw_graph_edges(
    ax: plt.axis,
    pos: dict,
    graph: nx.Graph,
    edges: list[tuple] | None = None,
    normal_node_size: float = NORMAL_NODE_SIZE,
) -> None:
    """Draws black lines between the nodes.

    Args:
        ax: The axis to draw the edges on.
        pos: The positions of the nodes.
        graph: The graph to draw the edges on.
        edges: The edges to draw. If ``None`` (default), all edges are drawn.
        normal_node_size: The size of the nodes. Defaults to ``NORMAL_NODE_SIZE``.
    """
    for u, v in graph.edges:
        if edges is not None and (u, v) not in edges and (v, u) not in edges:
            continue

        u_pos = pos[u]
        v_pos = pos[v]

        direction = v_pos - u_pos
        direction = direction / np.linalg.norm(direction)

        start_point = u_pos + direction * normal_node_size / 2
        end_point = v_pos - direction * normal_node_size / 2

        connection = mpath.Path(
            [start_point, end_point],
            [mpath.Path.MOVETO, mpath.Path.LINETO],
        )

        patch = mpatches.PathPatch(connection, facecolor="none", lw=1, edgecolor="black")
        ax.add_patch(patch)


def _draw_graph_labels(ax: plt.axis, pos: dict, graph: nx.Graph, nodes: list | None = None) -> None:
    """Adds labels to the nodes of the graph.

    Args:
        ax: The axis to draw the labels on.
        pos: The positions of the nodes.
        graph: The graph to draw the labels on.
        nodes: The nodes to draw the labels on. If ``None`` (default), all nodes are drawn.
    """
    for node in graph.nodes:
        if nodes is not None and node not in nodes:
            continue
        label = graph.nodes.get(node)["label"]
        position = pos[node]
        ax.text(
            position[0],
            position[1],
            label,
            fontsize=plt.rcParams["font.size"] + 1,
            ha="center",
            va="center",
            color="black",
        )


def _adjust_position(
    pos: dict, graph: nx.Graph, normal_node_size: float = NORMAL_NODE_SIZE
) -> dict:
    """Moves the nodes in the graph further apart if they are too close together."""
    # get the minimum distance between two nodes
    min_distance = 1e10
    for u, v in graph.edges:
        distance = np.linalg.norm(pos[u] - pos[v])
        min_distance = min(min_distance, distance)

    # adjust the positions if the nodes are too close together
    min_edge_distance = normal_node_size + normal_node_size / 2
    if min_distance < min_edge_distance:
        for node in pos:
            pos[node] = pos[node] * min_edge_distance / min_distance

    return pos


def si_graph_plot(
    interaction_values: shapiq.InteractionValues,
    graph: list[tuple] | nx.Graph | None = None,
    n_interactions: int | None = None,
    draw_threshold: float = 0.0,
    random_seed: int = 42,
    size_factor: float = 1.0,
    plot_explanation: bool = True,
    compactness: float = 1e10,
    feature_names: list | None = None,
    cubic_scaling: bool = False,
    pos: dict | None = None,
    node_size_scaling: float = 1.0,
    min_max_interactions: tuple[float, float] | None = None,
    adjust_node_pos: bool = False,
    spring_k: float | None = None,
    interaction_direction: str | None = None,
    node_area_scaling: bool = False,
    draw_original: bool = False,  # ADDED in this version
    figsize = (7, 7),
    show: bool = False,
) -> tuple[plt.figure, plt.axis, dict, dict] | None:
    """Plots the interaction values as an explanation graph.

    Args:
        interaction_values: The interaction values to plot.
        graph: The underlying graph structure as a list of edge tuples or a networkx graph. If a
            networkx graph is provided, the nodes are used as the players and the edges are used as
            the connections between the players. Defaults to ``None``, which creates a graph with
            all nodes from the interaction values without any edges between them.
        n_interactions: The number of interactions to plot. If ``None``, all interactions are plotted
            according to the draw_threshold.
        draw_threshold: The threshold to draw an edge (i.e. only draw explanations with an
            interaction value higher than this threshold).
        random_seed: The random seed to use for layout of the graph.
        size_factor: The factor to scale the explanations by (a higher value will make the
            interactions and main effects larger). Defaults to ``1.0``.
        plot_explanation: Whether to plot the explanation or only the original graph. Defaults to
            ``True``.
        compactness: A scaling factor for the underlying spring layout. A higher compactness value
            will move the interactions closer to the graph nodes. If your graph looks weird, try
            adjusting this value, e.g. ``[0.1, 1.0, 10.0, 100.0, 1000.0]``. Defaults to ``1.0``.
        feature_names: A list of feature names to use for the nodes in the graph. If ``None``,
            the feature indices are used instead. Defaults to ``None``.
        cubic_scaling: Whether to scale the size of explanations cubically (``True``) or linearly
            (``False``, default). Cubic scaling puts more emphasis on larger interactions in the plot.
            Defaults to ``False``.
        pos: The positions of the nodes in the graph. If ``None``, the spring layout is used to
            position the nodes. Defaults to ``None``.
        node_size_scaling: The scaling factor for the node sizes. This can be used to make the nodes
            larger or smaller depending on how the graph looks. Defaults to ``1.0`` (no scaling).
            Negative values will make the nodes smaller, positive values will make the nodes larger.
        min_max_interactions: The minimum and maximum interaction values to use for scaling the
            interactions as a tuple ``(min, max)``. If ``None``, the minimum and maximum interaction
            values are used. Defaults to ``None``.
        adjust_node_pos: Whether to adjust the node positions such that the nodes are at least
            ``NORMAL_NODE_SIZE`` apart. Defaults to ``False``.
        spring_k: The spring constant for the spring layout. If `None`, the spring constant is
            calculated based on the number of nodes in the graph. Defaults to ``None``.
        interaction_direction: The sign of the interaction values to plot. If ``None``, all
            interactions are plotted. Possible values are ``"positive"`` and
            ``"negative"``. Defaults to ``None``.
        node_area_scaling: Whether to scale the node sizes based on the area of the nodes (``True``)
             or the radius of the nodes (``False``). Defaults to ``False``.
        show: Whether to show or return the plot. Defaults to ``False``.

    Returns:
        The figure and axis of the plot if ``show`` is ``False``. Otherwise, ``None``.
    """

    normal_node_size = NORMAL_NODE_SIZE * node_size_scaling
    base_size = BASE_SIZE * node_size_scaling

    label_mapping = None
    if feature_names is not None:
        label_mapping = {i: feature_names[i] for i in range(len(feature_names))}

    # fill the original graph with the edges and nodes
    if isinstance(graph, nx.Graph):
        original_graph = graph
        graph_nodes = list(original_graph.nodes)
        # check if graph has labels
        if "label" not in original_graph.nodes[graph_nodes[0]]:
            for node in graph_nodes:
                node_label = label_mapping.get(node, node) if label_mapping is not None else node
                original_graph.nodes[node]["label"] = node_label
    elif isinstance(graph, list):
        original_graph, graph_nodes = nx.Graph(), []
        for edge in graph:
            original_graph.add_edge(*edge)
            nodel_labels = [edge[0], edge[1]]
            if label_mapping is not None:
                nodel_labels = [label_mapping.get(node, node) for node in nodel_labels]
            original_graph.add_node(edge[0], label=nodel_labels[0])
            original_graph.add_node(edge[1], label=nodel_labels[1])
            graph_nodes.extend([edge[0], edge[1]])
    else:  # graph is considered None
        original_graph = nx.Graph()
        graph_nodes = list(range(interaction_values.n_players))
        for node in graph_nodes:
            node_label = label_mapping.get(node, node) if label_mapping is not None else node
            original_graph.add_node(node, label=node_label)

    if n_interactions is not None:
        # get the top n interactions
        interaction_values = interaction_values.get_top_k(n_interactions)

    # get the interactions to plot (sufficiently large)
    interactions_to_plot = {}
    min_interaction, max_interaction = 1e10, 0.0
    for interaction, interaction_pos in interaction_values.interaction_lookup.items():
        if len(interaction) == 0:
            continue
        interaction_value = interaction_values.values[interaction_pos]
        min_interaction = min(abs(interaction_value), min_interaction)
        max_interaction = max(abs(interaction_value), max_interaction)
        if abs(interaction_value) > draw_threshold:
            if interaction_direction == "positive" and interaction_value < 0:
                continue
            if interaction_direction == "negative" and interaction_value > 0:
                continue
            interactions_to_plot[interaction] = interaction_value

    if min_max_interactions is not None:
        min_interaction, max_interaction = min_max_interactions

    # create explanation graph
    explanation_graph, explanation_nodes, explanation_edges = nx.Graph(), [], []
    explanation_attributes = {}
    for interaction, interaction_value in interactions_to_plot.items():
        interaction_size = len(interaction)
        interaction_strength = abs(interaction_value)

        attributes = {
            "color": get_color(interaction_value),
            "alpha": _normalize_value(
                interaction_value, max_interaction, BASE_ALPHA_VALUE, cubic_scaling
            ),
            "interaction": interaction,
            "weight": interaction_strength * compactness,
            "size": _normalize_value(
                interaction_value, max_interaction, base_size * size_factor, cubic_scaling
            ),
        }
        explanation_attributes[interaction] = attributes

        # add main effect explanations as nodes
        if interaction_size == 1:
            player = interaction[0]
            explanation_graph.add_node(player, **attributes)
            explanation_nodes.append(player)

        # add 2-way interaction explanations as edges
        if interaction_size >= 2:
            explanation_edges.append(interaction)
            player_last = interaction[-1]
            if interaction_size > 2:
                dummy_node = tuple(interaction)
                explanation_graph.add_node(dummy_node, **attributes)
                player_last = dummy_node
            # add the edges between the players
            for player in interaction[:-1]:
                explanation_graph.add_edge(player, player_last, **attributes)

    # position first the original graph structure
    if pos is None:
        pos = nx.spring_layout(original_graph, seed=random_seed, k=spring_k)
        pos = nx.kamada_kawai_layout(original_graph, scale=1.0, pos=pos)
    else:
        # pos is given but we need to scale the positions potentially
        min_pos = np.min(list(pos.values()), axis=0)
        max_pos = np.max(list(pos.values()), axis=0)
        pos = {node: (pos[node] - min_pos) / (max_pos - min_pos) for node in pos}

    # adjust pos such that the nodes are at least NORMAL_NODE_SIZE apart
    if adjust_node_pos:
        pos = _adjust_position(pos, original_graph)

    # create the plot
    fig, ax = plt.subplots(figsize=figsize)
    if plot_explanation:
        # position now again the hyper-edges onto the normal nodes weight param is weight
        pos_explain = nx.spring_layout(
            explanation_graph, weight="weight", seed=random_seed, pos=pos, fixed=graph_nodes
        )
        pos.update(pos_explain)
        _draw_fancy_hyper_edges(ax, pos, explanation_graph, hyper_edges=explanation_edges)
        _draw_explanation_nodes(
            ax,
            pos,
            explanation_graph,
            nodes=explanation_nodes,
            normal_node_size=normal_node_size,
            node_area_scaling=node_area_scaling,
        )

    # add the original graph structure on top
    if draw_original:
        _draw_graph_nodes(ax, pos, original_graph, normal_node_size=normal_node_size)
        _draw_graph_edges(ax, pos, original_graph, normal_node_size=normal_node_size)
        _draw_graph_labels(ax, pos, original_graph)

    # tidy up the plot
    # ax.set_aspect("equal", adjustable="datalim")  # make y- and x-axis scales equal
    ax.axis("off")  # remove axis

    if not show:
        return fig, ax, pos, explanation_attributes
    plt.show()
