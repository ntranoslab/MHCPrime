import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm

def plot_bars_or_boxplots(
    df,
    model_col,
    score_col,
    model_groups,
    x_ticks,
    colors=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font=None,
    bar_width=0.8,
    group_spacing=0.3,
    bar_spacing=0.0,
    bar_outline_color='black',
    bar_outline_width=1.0,
    mode='bar',
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    pval_df=None,
    pval_comparisons=None,
    pval_pairs=None,
    pval_base_only=False,
    pval_base_col=None,
    pval_merge_nonoverlapping=True,
    pval_format='stars',
    pval_fontsize=5,
    pval_linewidth=0.35,
    pval_color='black',
    pval_bar_offset=0.035,
    pval_bar_spacing=0.055,
    pval_text_offset=0.004,
    pval_star_text_offset=0.001,
    pval_ns_text_offset=0.004,
    reference_hlines=None,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    x_margin=None,
    xtick_rotation=0,
    edge_margin=None,
    legend_mode='separate',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    ax=None,
    show=True,
    save_path=None,
):
    matplotlib.rcParams['pdf.fonttype'] = 42
    if default_font is not None:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [default_font, 'Arial', 'Helvetica', 'DejaVu Sans']

    font_kw = {} if default_font is None else {'fontfamily': default_font}

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'title_y': 1.04,
        'title_color': 'black',
        'title_bbox': None,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    if x_ticks is None or len(x_ticks) == 0:
        hide_x_ticks = True
        n_orig = max(len(model_list) for model_list in model_groups.values())
        x_ticks_working = [''] * n_orig
    else:
        hide_x_ticks = all(str(tick) == '' for tick in x_ticks)
        x_ticks_working = x_ticks

    def _pval_to_stars(p):
        if isinstance(p, str):
            p_str = p.strip()
            if p_str in {'ns', '*', '**', '***', '****'}:
                return p_str
            try:
                p = float(p_str)
            except ValueError:
                return p_str

        if pd.isna(p):
            return None

        p = float(p)

        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    hline_col_set = set(reference_hlines.keys())

    hline_entries = []
    for col_name, style in reference_hlines.items():
        if mode == 'box':
            if col_name in df.columns:
                agg = style.get('agg', 'mean')
                if agg == 'median':
                    y_val = df[col_name].dropna().median()
                else:
                    y_val = df[col_name].dropna().mean()
            else:
                continue
        else:
            row = df[df[model_col] == col_name]
            if len(row) > 0:
                y_val = row[score_col].values[0]
            else:
                continue
        hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for label, model_list in model_groups.items():
        filtered = []
        for col in model_list:
            if col in hline_col_set:
                filtered.append(None)
            else:
                filtered.append(col)
        filtered_groups[label] = filtered

    n_orig = len(x_ticks_working)
    keep_ticks = []
    for tick_idx in range(n_orig):
        has_any = False
        for label, flist in filtered_groups.items():
            if tick_idx < len(flist) and flist[tick_idx] is not None:
                has_any = True
                break
        if has_any:
            keep_ticks.append(tick_idx)

    x_ticks_final = [x_ticks_working[i] for i in keep_ticks]
    final_groups = {}
    for label, flist in filtered_groups.items():
        final_groups[label] = [flist[i] for i in keep_ticks if i < len(flist)]

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(model_groups.keys())
        }

    created_own_ax = ax is None
    if created_own_ax:
        fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])
    else:
        fig = ax.figure

    n_groups = len(x_ticks_final)

    bars_per_position = [
        sum(
            1 for label, model_list in final_groups.items()
            if tick_idx < len(model_list) and model_list[tick_idx] is not None
        )
        for tick_idx in range(n_groups)
    ]

    cluster_widths = [
        max(n, 1) * bar_width + max(n - 1, 0) * bar_spacing
        for n in bars_per_position
    ]

    group_positions = np.zeros(n_groups)
    cursor = 0.0
    for i, cw in enumerate(cluster_widths):
        group_positions[i] = cursor + cw / 2.0
        cursor += cw + group_spacing

    edge_margin = x_margin if x_margin is not None else group_spacing / 2.0
    ax.set_xlim(
        group_positions[0] - cluster_widths[0] / 2.0 - edge_margin,
        group_positions[-1] + cluster_widths[-1] / 2.0 + edge_margin,
    )

    plotted_positions = {}
    plotted_upper_whiskers = {}

    def _upper_boxplot_whisker(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        valid = vals[vals <= upper_limit]

        if len(valid) == 0:
            return np.max(vals)

        return np.max(valid)

    for group_idx, (label, model_list) in enumerate(final_groups.items()):
        color = colors[label]

        if mode == 'bar':
            scores = []
            for col_name in model_list:
                if col_name is None:
                    scores.append(None)
                    continue

                score = df[df[model_col] == col_name][score_col].values
                scores.append(score[0] if len(score) > 0 else 0)

            first_plotted = True
            for tick_idx in range(len(scores)):
                if tick_idx >= n_groups or scores[tick_idx] is None:
                    continue

                col_name = model_list[tick_idx]
                if col_name is None:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos
                plotted_upper_whiskers[(tick_idx, col_name)] = scores[tick_idx]

                ax.bar(
                    x_pos,
                    scores[tick_idx],
                    width=bar_width,
                    color=color,
                    edgecolor=bar_outline_color,
                    linewidth=bar_outline_width,
                    label=label if first_plotted else "",
                    zorder=2,
                )
                first_plotted = False

        elif mode == 'box':
            first_plotted = True
            for tick_idx, col_name in enumerate(model_list):
                if tick_idx >= n_groups or col_name is None:
                    continue

                if col_name not in df.columns:
                    continue

                vals = df[col_name].dropna().values.tolist()
                if not vals:
                    continue

                cw = cluster_widths[tick_idx]
                left_edge = group_positions[tick_idx] - cw / 2.0
                x_pos = left_edge + group_idx * (bar_width + bar_spacing) + bar_width / 2.0

                plotted_positions[(tick_idx, col_name)] = x_pos
                plotted_upper_whiskers[(tick_idx, col_name)] = _upper_boxplot_whisker(vals)

                flier_c = box_flier_color if box_flier_color is not None else color

                ax.boxplot(
                    [vals],
                    positions=[x_pos],
                    widths=bar_width * 0.9,
                    patch_artist=True,
                    notch=box_notch,
                    manage_ticks=False,
                    zorder=2,
                    flierprops=dict(
                        marker=box_flier_marker,
                        markersize=box_flier_size,
                        markerfacecolor=flier_c,
                        markeredgecolor=flier_c,
                        alpha=box_flier_alpha,
                    ),
                    medianprops=dict(
                        color=box_median_color,
                        linewidth=box_median_width,
                    ),
                    whiskerprops=dict(
                        color=bar_outline_color,
                        linewidth=box_whisker_width,
                        linestyle=box_whisker_style,
                    ),
                    capprops=dict(
                        color='none',
                        linewidth=0,
                    ),
                    boxprops=dict(
                        facecolor=color,
                        edgecolor=bar_outline_color,
                        linewidth=bar_outline_width,
                        alpha=box_patch_alpha,
                    ),
                )
                first_plotted = False

    if pval_df is not None and mode in ['box', 'bar']:
        for tick_idx in range(n_groups):
            cols_at_tick = []
            for label, model_list in final_groups.items():
                if tick_idx < len(model_list):
                    col_name = model_list[tick_idx]
                    if col_name is not None and (tick_idx, col_name) in plotted_positions:
                        cols_at_tick.append(col_name)

            if len(cols_at_tick) < 2:
                continue

            comparison_source = pval_pairs if pval_pairs is not None else pval_comparisons

            if comparison_source is None:
                if pval_base_only:
                    if pval_base_col is None:
                        raise ValueError(
                            "pval_base_col must be provided when pval_base_only=True."
                        )
                    if pval_base_col not in cols_at_tick:
                        comparisons = []
                    else:
                        comparisons = [
                            (pval_base_col, col)
                            for col in cols_at_tick
                            if col != pval_base_col
                        ]
                else:
                    comparisons = [
                        (cols_at_tick[i], cols_at_tick[j])
                        for i in range(len(cols_at_tick) - 1)
                        for j in range(i + 1, len(cols_at_tick))
                    ]
            else:
                comparisons = comparison_source

            upper_vals = [
                plotted_upper_whiskers[(tick_idx, col)]
                for col in cols_at_tick
                if (tick_idx, col) in plotted_upper_whiskers
            ]

            if len(upper_vals) == 0:
                continue

            base_y = np.nanmax(upper_vals) + pval_bar_offset

            pval_levels = []
            hierarchical_counter = 0

            def _intervals_overlap(interval_a, interval_b, eps=1e-12):
                a1, a2 = interval_a
                b1, b2 = interval_b
                return max(a1, b1) < min(a2, b2) - eps

            for col_a, col_b in comparisons:
                if (tick_idx, col_a) not in plotted_positions:
                    continue
                if (tick_idx, col_b) not in plotted_positions:
                    continue

                x_a = plotted_positions[(tick_idx, col_a)]
                x_b = plotted_positions[(tick_idx, col_b)]

                if x_a <= x_b:
                    x1, x2 = x_a, x_b
                else:
                    x1, x2 = x_b, x_a

                pval_value = None

                if col_a in pval_df.index and col_b in pval_df.columns:
                    pval_value = pval_df.loc[col_a, col_b]
                elif col_b in pval_df.index and col_a in pval_df.columns:
                    pval_value = pval_df.loc[col_b, col_a]

                if pval_value is None:
                    continue
                if pd.isna(pval_value):
                    continue

                if pval_format == 'stars':
                    pval_text = _pval_to_stars(pval_value)
                else:
                    pval_text = str(pval_value)

                if pval_text is None:
                    continue
                if str(pval_text) == '':
                    continue

                current_interval = (x1, x2)

                if pval_merge_nonoverlapping:
                    assigned_level = None

                    for level_idx, occupied_intervals in enumerate(pval_levels):
                        has_overlap = any(
                            _intervals_overlap(current_interval, existing_interval)
                            for existing_interval in occupied_intervals
                        )

                        if not has_overlap:
                            assigned_level = level_idx
                            break

                    if assigned_level is None:
                        assigned_level = len(pval_levels)
                        pval_levels.append([])

                    pval_levels[assigned_level].append(current_interval)

                else:
                    assigned_level = hierarchical_counter
                    hierarchical_counter += 1

                y = base_y + assigned_level * pval_bar_spacing

                ax.plot(
                    [x1, x2],
                    [y, y],
                    color=pval_color,
                    linewidth=pval_linewidth,
                    solid_capstyle='butt',
                    zorder=3,
                )

                if str(pval_text).lower() == 'ns':
                    text_offset = pval_ns_text_offset
                elif set(str(pval_text)) <= {'*'}:
                    text_offset = pval_star_text_offset
                else:
                    text_offset = pval_text_offset

                ax.text(
                    (x1 + x2) / 2.0,
                    y + text_offset,
                    str(pval_text),
                    ha='center',
                    va='bottom',
                    fontsize=pval_fontsize,
                    color=pval_color,
                    zorder=4,
                    **font_kw,
                )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            label=style.get('label', col_name),
            zorder=1,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None

        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=line_label,
            zorder=1,
        )

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            x_range = group_positions[-1] - group_positions[0] if len(group_positions) > 1 else 1
            text_x = group_positions[0] + (x_range * hline_text_x_offset)

            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'

            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                va=va,
                ha='left',
                zorder=3,
                **font_kw,
            )

    if hide_x_ticks:
        ax.set_xticks([])
    else:
        ax.set_xticks(group_positions)

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        labelpad=params['xlabel_pad'],
        **font_kw,
    )
    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        labelpad=params['ylabel_pad'],
        **font_kw,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length'] + 2,
    )
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if not hide_x_ticks:
        ax.set_xticklabels(x_ticks_final, rotation=xtick_rotation, ha='center')

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    if default_font is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    for lbl_key in model_groups.keys():
        c = colors[lbl_key]
        handles.append(
            mpatches.Patch(
                facecolor=c,
                edgecolor=bar_outline_color,
                linewidth=bar_outline_width,
            )
        )
        labels_list.append(lbl_key)

    for y_val, style, col_name in hline_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get('color', 'gray'),
                linestyle=style.get('linestyle', '--'),
                linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                alpha=style.get('alpha', 1.0),
            )
        )
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(
            Line2D(
                [0],
                [0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width * params['legend_handle_size'],
            )
        )
        labels_list.append(hline_label)

    if legend_mode == 'integrated':
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        legend = fig.legend(
            handles,
            labels_list,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            ncol=legend_ncol,
            frameon=False,
            fontsize=params['legend_fontsize'],
        )
        if default_font is not None:
            for text in legend.get_texts():
                text.set_fontfamily(default_font)

    if created_own_ax:
        plt.tight_layout()

    if params.get('title', '') not in ['', None]:
        title_weight = 'bold' if params['title_bold'] else 'normal'

        ax.text(
            0.5,
            params.get('title_y', 1.04),
            params['title'],
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=params['title_fontsize'],
            fontweight=title_weight,
            color=params.get('title_color', 'black'),
            bbox=params.get('title_bbox', None),
            clip_on=False,
            zorder=100,
            **font_kw,
        )

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''

        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            dpi=300,
            bbox_inches='tight',
        )

    if show:
        plt.show()

    if legend_mode == 'separate' and created_own_ax:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(
                params['legend_handle_size'] * 2,
                n_items * 0.4 * params['legend_handle_size'],
            ),
            dpi=params['dpi'],
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        legend = ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=legend_ncol,
        )
        if default_font is not None:
            for text in legend.get_texts():
                text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                dpi=300,
            )

        if show:
            plt.show()

    return fig, ax


def plot_bars_or_boxplots_grid(
    plot_dict,
    save_path=None,
    panel_gap=0.0,
    grid_dpi=300,
    legend_dict=None,
    legend_fontsize=10,
    legend_ncol=None,
    legend_outline_color='black',
    legend_outline_width=0.8,
    legend_swatch_size=1.2,
    legend_handletextpad=0.5,
    legend_columnspacing=1.0,
    legend_labelspacing=0.4,
    legend_height=0.5,
    legend_y=None,
    **base_plotting_parameters,
):
    grid_font = base_plotting_parameters.get('default_font', None)

    matplotlib.rcParams['pdf.fonttype'] = 42
    if grid_font is not None:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [grid_font, 'Arial', 'Helvetica', 'DejaVu Sans']

    font_kw = {} if grid_font is None else {'fontfamily': grid_font}

    base_plot_params = base_plotting_parameters.get('plot_params', {}).copy()
    single_figsize = base_plot_params.get('figsize', (1.20, 1.65))
    single_w, single_h = single_figsize

    has_legend = legend_dict is not None and len(legend_dict) > 0
    legend_reserve = legend_height if has_legend else 0.0

    n_panels = len(plot_dict)
    fig_w = n_panels * single_w + max(n_panels - 1, 0) * panel_gap
    fig_h = single_h + legend_reserve

    # Probe the single-panel axes location after tight_layout, without title.
    probe_fig, probe_ax = plt.subplots(figsize=single_figsize, dpi=grid_dpi)

    probe_ax.set_xlabel(
        base_plot_params.get('xlabel', ''),
        fontsize=base_plot_params.get('xlabel_fontsize', 12),
        labelpad=base_plot_params.get('xlabel_pad', 4.0),
        **font_kw,
    )
    probe_ax.set_ylabel(
        base_plot_params.get('ylabel', ''),
        fontsize=base_plot_params.get('ylabel_fontsize', 12),
        labelpad=base_plot_params.get('ylabel_pad', 4.0),
        **font_kw,
    )

    probe_ax.set_ylim(base_plotting_parameters.get('ylim', (0.0, 1.0)))
    probe_ax.set_xticks([])

    probe_ax.tick_params(
        axis='both',
        which='major',
        length=base_plot_params.get('tick_length', 4.0),
        width=base_plot_params.get('axis_linewidth', 0.8),
        direction='out',
        pad=base_plot_params.get('tick_length', 4.0) + 2,
    )
    probe_ax.tick_params(axis='x', labelsize=base_plot_params.get('xtick_labelsize', 10))
    probe_ax.tick_params(axis='y', labelsize=base_plot_params.get('ytick_labelsize', 10))

    plt.figure(probe_fig.number)
    plt.tight_layout()
    probe_pos = probe_ax.get_position()
    plt.close(probe_fig)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=grid_dpi)
    axes = []

    for i, (panel_key, panel_info) in enumerate(plot_dict.items()):
        slot_left = i * (single_w + panel_gap)

        ax_left = (slot_left + probe_pos.x0 * single_w) / fig_w
        ax_bottom = (legend_reserve + probe_pos.y0 * single_h) / fig_h
        ax_width = (probe_pos.width * single_w) / fig_w
        ax_height = (probe_pos.height * single_h) / fig_h

        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
        axes.append(ax)

        panel_params = base_plotting_parameters.copy()

        panel_plot_params = base_plot_params.copy()
        panel_plot_params['title'] = panel_info.get('title', panel_key)

        if 'plot_params' in panel_info:
            panel_plot_params.update(panel_info['plot_params'])

        panel_params['plot_params'] = panel_plot_params
        panel_params['pval_df'] = panel_info.get('pval_df', None)

        if 'pval_comparisons' in panel_info:
            panel_params['pval_comparisons'] = panel_info['pval_comparisons']

        plot_bars_or_boxplots(
            df=panel_info['df'],
            ax=ax,
            show=False,
            save_path=None,
            **panel_params,
        )

    if has_legend:
        legend_handles = [
            mpatches.Patch(
                facecolor=color,
                edgecolor=legend_outline_color,
                linewidth=legend_outline_width,
            )
            for color in legend_dict.values()
        ]
        legend_labels = list(legend_dict.keys())

        ncol = legend_ncol if legend_ncol is not None else len(legend_handles)
        anchor_y = legend_y if legend_y is not None else (legend_reserve / fig_h)

        leg = fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, anchor_y),
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=legend_swatch_size,
            handleheight=legend_swatch_size,
            handletextpad=legend_handletextpad,
            columnspacing=legend_columnspacing,
            labelspacing=legend_labelspacing,
        )
        if grid_font is not None:
            for text in leg.get_texts():
                text.set_fontfamily(grid_font)

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", format='pdf', bbox_inches='tight', dpi=grid_dpi)
        fig.savefig(f"{save_path}.png", format='png', bbox_inches='tight', dpi=grid_dpi)

    plt.show()

    return fig, axes

def plot_grouped_boxplots(
    groups,
    colors=None,
    legend=None,
    plot_params=None,
    ylim=None,
    grid_alpha=0.0,
    default_font=None,
    box_width=0.6,
    group_spacing=1.0,
    box_spacing=0.05,
    box_outline_color='black',
    box_outline_width=1.0,
    box_flier_marker='o',
    box_flier_size=3,
    box_flier_color=None,
    box_flier_alpha=0.6,
    box_flier_edge_width=0,
    box_median_color='black',
    box_median_width=1.5,
    box_whisker_style='-',
    box_whisker_width=1.0,
    box_notch=False,
    box_patch_alpha=0.9,
    sort_by=None,
    sort_ascending=False,
    hline_y=None,
    hline_label=None,
    hline_color='gray',
    hline_width=1.0,
    hline_style=':',
    hline_text=None,
    hline_text_position='above',
    hline_text_color=None,
    hline_text_fontsize=10,
    hline_text_offset=0.01,
    hline_text_x_offset=0.05,
    hline_in_legend=True,
    reference_hlines=None,
    legend_mode='integrated',
    legend_loc='lower center',
    legend_bbox=None,
    legend_ncol=None,
    save_path=None,
    compute_pvalues=False,
    pvalue_test='wilcoxon',
    cluster_col=None,
    pval_df=None,
    pval_ref_col=None,
    pval_ref_stars_only=True,
    pval_format='stars',
    pval_fontsize=8,
    pval_color='black',
    pval_star_y=None,
    pval_star_y_frac=0.985,
    pval_star_text_offset=0.0,
):
    matplotlib.rcParams['pdf.fonttype'] = 42
    if default_font is not None:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [default_font, 'Arial', 'Helvetica', 'DejaVu Sans']

    font_kw = {} if default_font is None else {'fontfamily': default_font}

    default_params = {
        'title': '',
        'title_fontsize': 14,
        'title_bold': True,
        'xlabel': '',
        'ylabel': '',
        'xlabel_fontsize': 12,
        'ylabel_fontsize': 12,
        'figsize': (10, 6),
        'dpi': 100,
        'legend_fontsize': 10,
        'legend_cols': 1,
        'legend_handle_size': 1.0,
        'axis_linewidth': 0.8,
        'tick_length': 4.0,
        'xlabel_pad': 4.0,
        'ylabel_pad': 4.0,
        'xtick_labelsize': 10,
        'ytick_labelsize': 10,
        'xtick_rotation': 0,
        'title_y': None,
        'title_pad': 6.0,
    }
    if plot_params is not None:
        default_params.update(plot_params)
    params = default_params

    if legend_ncol is None:
        legend_ncol = params['legend_cols']

    if reference_hlines is None:
        reference_hlines = {}

    hline_col_set = set(reference_hlines.keys())

    def _normalize_group(group_value):
        if isinstance(group_value, list):
            tuples = group_value
        else:
            tuples = [group_value]
        pairs = []
        for t in tuples:
            df_g = t[0]
            cols = t[1:]
            for c in cols:
                pairs.append((df_g, c))
        return pairs

    def _pval_to_stars(p):
        if isinstance(p, str):
            p_str = p.strip()
            if p_str in {'ns', '*', '**', '***', '****'}:
                return p_str
            try:
                p = float(p_str)
            except ValueError:
                return p_str

        if pd.isna(p):
            return None

        p = float(p)

        if p <= 1e-4:
            return '****'
        elif p <= 1e-3:
            return '***'
        elif p <= 1e-2:
            return '**'
        elif p <= 0.05:
            return '*'
        else:
            return 'ns'

    def _lookup_symmetric_pval(pval_table, col_a, col_b):
        if pval_table is None:
            return None

        if col_a in pval_table.index and col_b in pval_table.columns:
            return pval_table.loc[col_a, col_b]
        elif col_b in pval_table.index and col_a in pval_table.columns:
            return pval_table.loc[col_b, col_a]

        return None

    all_cols = set()
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            all_cols.add(col)

    default_colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]
    if colors is None:
        colors = {}
        for i, col in enumerate(sorted(all_cols)):
            colors[col] = default_colors[i % len(default_colors)]

    hline_entries = []
    hline_vals_accum = {}
    for tick_label, group_value in groups.items():
        for df_g, col in _normalize_group(group_value):
            if col in hline_col_set and col in df_g.columns:
                if col not in hline_vals_accum:
                    hline_vals_accum[col] = []
                hline_vals_accum[col].extend(df_g[col].dropna().values.tolist())

    for col_name, vals in hline_vals_accum.items():
        if vals:
            style = reference_hlines[col_name]
            agg = style.get('agg', 'mean')
            if agg == 'median':
                y_val = np.median(vals)
            else:
                y_val = np.mean(vals)
            hline_entries.append((y_val, style, col_name))

    filtered_groups = {}
    for tick_label, group_value in groups.items():
        pairs = [
            (df_g, col)
            for df_g, col in _normalize_group(group_value)
            if col not in hline_col_set
        ]
        if pairs:
            filtered_groups[tick_label] = pairs

    if sort_by is not None:
        def _group_stat(pairs):
            vals = []
            for df_g, col in pairs:
                if col in df_g.columns:
                    vals.extend(df_g[col].dropna().values.tolist())
            if not vals:
                return np.nan
            return np.median(vals) if sort_by == 'median' else np.mean(vals)

        def _sort_key(item):
            stat = _group_stat(item[1])
            if np.isnan(stat):
                return np.inf if sort_ascending else -np.inf
            return stat

        filtered_groups = dict(
            sorted(
                filtered_groups.items(),
                key=_sort_key,
                reverse=not sort_ascending,
            )
        )

    max_cols = max(len(pairs) for pairs in filtered_groups.values()) if filtered_groups else 1
    n_groups = len(filtered_groups)

    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])

    group_width = max_cols * box_width + (max_cols - 1) * box_spacing
    group_centers = np.arange(n_groups) * (group_width + group_spacing)

    tick_labels = []

    plotted_positions = {}
    plotted_upper_whiskers = {}

    def _upper_boxplot_whisker(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        valid = vals[vals <= upper_limit]

        if len(valid) == 0:
            return np.max(vals)

        return np.max(valid)

    for gi, (tick_label, pairs) in enumerate(filtered_groups.items()):
        tick_labels.append(tick_label)
        n_cols = len(pairs)

        total_box_width = n_cols * box_width + (n_cols - 1) * box_spacing
        start = group_centers[gi] - total_box_width / 2 + box_width / 2

        for ci, (df_g, col) in enumerate(pairs):
            if col not in df_g.columns:
                continue
            vals = df_g[col].dropna().values.tolist()
            if not vals:
                continue

            x_pos = start + ci * (box_width + box_spacing)
            color = colors.get(col, default_colors[0])
            flier_c = box_flier_color if box_flier_color is not None else color

            plotted_positions[(tick_label, col)] = x_pos
            plotted_upper_whiskers[(tick_label, col)] = _upper_boxplot_whisker(vals)

            bp = ax.boxplot(
                [vals],
                positions=[x_pos],
                widths=box_width * 0.9,
                patch_artist=True,
                notch=box_notch,
                manage_ticks=False,
                zorder=2,
                flierprops=dict(
                    marker=box_flier_marker,
                    markersize=box_flier_size,
                    markerfacecolor=flier_c,
                    markeredgecolor=flier_c,
                    markeredgewidth=box_flier_edge_width,
                    alpha=box_flier_alpha,
                ),
                medianprops=dict(
                    color=box_median_color,
                    linewidth=box_median_width,
                ),
                whiskerprops=dict(
                    color=box_outline_color,
                    linewidth=box_whisker_width,
                    linestyle=box_whisker_style,
                ),
                capprops=dict(
                    color='none',
                    linewidth=0,
                ),
                boxprops=dict(
                    facecolor=color,
                    edgecolor=box_outline_color,
                    linewidth=box_outline_width,
                    alpha=box_patch_alpha,
                ),
            )

    for y_val, style, col_name in hline_entries:
        ax.axhline(
            y=y_val,
            color=style.get('color', 'gray'),
            linestyle=style.get('linestyle', '--'),
            linewidth=style.get('linewidth', 1.5),
            alpha=style.get('alpha', 1.0),
            zorder=3,
        )

    if hline_y is not None:
        line_label = hline_label if hline_in_legend else None
        ax.axhline(
            y=hline_y,
            color=hline_color,
            linestyle=hline_style,
            linewidth=hline_width,
            label=line_label,
            zorder=3,
        )

        if hline_text is not None:
            text_color = hline_text_color if hline_text_color is not None else hline_color
            if n_groups > 1:
                x_range = group_centers[-1] - group_centers[0]
            else:
                x_range = 1
            text_x = group_centers[0] + (x_range * hline_text_x_offset)
            if hline_text_position == 'above':
                text_y = hline_y + hline_text_offset
                va = 'bottom'
            else:
                text_y = hline_y - hline_text_offset
                va = 'top'
            ax.text(
                text_x,
                text_y,
                hline_text,
                color=text_color,
                fontsize=hline_text_fontsize,
                va=va,
                ha='left',
                zorder=3,
                **font_kw,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        tick_labels,
        rotation=params['xtick_rotation'],
        ha='right' if params['xtick_rotation'] != 0 else 'center',
    )

    ax.set_xlabel(
        params['xlabel'],
        fontsize=params['xlabel_fontsize'],
        labelpad=params['xlabel_pad'],
        **font_kw,
    )
    ax.set_ylabel(
        params['ylabel'],
        fontsize=params['ylabel_fontsize'],
        labelpad=params['ylabel_pad'],
        **font_kw,
    )

    title_weight = 'bold' if params['title_bold'] else 'normal'
    ax.set_title(
        params['title'],
        fontsize=params['title_fontsize'],
        fontweight=title_weight,
        y=params.get('title_y', None),
        pad=params.get('title_pad', 6.0),
        **font_kw,
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(params['axis_linewidth'])
    ax.spines['bottom'].set_linewidth(params['axis_linewidth'])

    ax.tick_params(
        axis='both',
        which='major',
        length=params['tick_length'],
        width=params['axis_linewidth'],
        direction='out',
        pad=params['tick_length'] + 2,
    )
    ax.tick_params(axis='x', labelsize=params['xtick_labelsize'])
    ax.tick_params(axis='y', labelsize=params['ytick_labelsize'])

    if ylim is not None:
        ax.set_ylim(ylim)

    if grid_alpha > 0:
        ax.grid(
            True,
            alpha=grid_alpha,
            linestyle='-',
            linewidth=0.5,
            axis='y',
            zorder=0,
        )

    if (
        pval_df is not None
        and pval_ref_col is not None
        and pval_ref_stars_only
        and len(plotted_positions) > 0
    ):
        y_low, y_high = ax.get_ylim()

        if pval_star_y is None:
            star_y_base = y_low + (y_high - y_low) * pval_star_y_frac
        else:
            star_y_base = pval_star_y

        star_y = star_y_base + pval_star_text_offset

        for (tick_label, col), x_pos in plotted_positions.items():
            if col == pval_ref_col:
                continue

            pval_value = _lookup_symmetric_pval(pval_df, pval_ref_col, col)

            if pval_value is None:
                continue
            if pd.isna(pval_value):
                continue

            if pval_format == 'stars':
                pval_text = _pval_to_stars(pval_value)
            else:
                pval_text = str(pval_value)

            if pval_text is None:
                continue
            if str(pval_text) == '':
                continue

            ax.text(
                x_pos,
                star_y,
                str(pval_text),
                ha='center',
                va='bottom',
                fontsize=pval_fontsize,
                color=pval_color,
                clip_on=False,
                zorder=10,
                **font_kw,
            )

    if default_font is not None:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontfamily(default_font)

    handles = []
    labels_list = []

    if legend is not None:
        for lbl_text, color in legend.items():
            handles.append(
                mpatches.Patch(
                    facecolor=color,
                    edgecolor=box_outline_color,
                    linewidth=box_outline_width,
                    alpha=box_patch_alpha,
                )
            )
            labels_list.append(lbl_text)

    for y_val, style, col_name in hline_entries:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get('color', 'gray'),
                linestyle=style.get('linestyle', '--'),
                linewidth=style.get('linewidth', 1.5) * params['legend_handle_size'],
                alpha=style.get('alpha', 1.0),
            )
        )
        labels_list.append(style.get('label', col_name))

    if hline_y is not None and hline_label is not None and hline_in_legend:
        handles.append(
            Line2D(
                [0],
                [0],
                color=hline_color,
                linestyle=hline_style,
                linewidth=hline_width * params['legend_handle_size'],
            )
        )
        labels_list.append(hline_label)

    if legend_mode == 'integrated' and handles:
        bbox = legend_bbox if legend_bbox is not None else (0.5, -0.08)
        leg = fig.legend(
            handles,
            labels_list,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            ncol=legend_ncol,
            frameon=False,
            fontsize=params['legend_fontsize'],
        )
        if default_font is not None:
            for text in leg.get_texts():
                text.set_fontfamily(default_font)

    plt.tight_layout()

    if save_path:
        suffix = '_plot' if legend_mode == 'separate' else ''
        fig.savefig(
            f"{save_path}{suffix}.pdf",
            format='pdf',
            bbox_inches='tight',
            dpi=300,
        )
        fig.savefig(
            f"{save_path}{suffix}.png",
            format='png',
            bbox_inches='tight',
            dpi=300,
        )

    plt.show()

    if legend_mode == 'separate' and handles:
        n_items = len(handles)
        fig_legend = plt.figure(
            figsize=(
                params['legend_handle_size'] * 2,
                n_items * 0.4 * params['legend_handle_size'],
            ),
            dpi=params['dpi'],
        )
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis('off')

        leg = ax_legend.legend(
            handles,
            labels_list,
            loc='center',
            frameon=False,
            fontsize=params['legend_fontsize'],
            ncol=legend_ncol,
        )
        if default_font is not None:
            for text in leg.get_texts():
                text.set_fontfamily(default_font)

        plt.tight_layout()

        if save_path:
            fig_legend.savefig(
                f"{save_path}_legend.pdf",
                format='pdf',
                bbox_inches='tight',
                dpi=300,
            )
            fig_legend.savefig(
                f"{save_path}_legend.png",
                format='png',
                dpi=300,
            )

        plt.show()

    if compute_pvalues:

        all_pairs = []
        for tick_label, group_value in groups.items():
            all_pairs.extend(_normalize_group(group_value))

        seen = set()
        unique_pairs = []
        for df_g, col in all_pairs:
            if col not in seen:
                seen.add(col)
                unique_pairs.append((df_g, col))

        col_names = [col for _, col in unique_pairs]
        n = len(col_names)
        pval_matrix = np.full((n, n), np.nan)

        for i in range(n):
            df_i, col_i = unique_pairs[i]
            for j in range(n):
                if i == j:
                    pval_matrix[i, j] = 1.0
                    continue
                if not np.isnan(pval_matrix[j, i]):
                    pval_matrix[i, j] = pval_matrix[j, i]
                    continue

                df_j, col_j = unique_pairs[j]

                if (
                    cluster_col is not None
                    and cluster_col in df_i.columns
                    and cluster_col in df_j.columns
                ):
                    merged = pd.merge(
                        df_i[[cluster_col, col_i]].dropna(),
                        df_j[[cluster_col, col_j]].dropna(),
                        on=cluster_col,
                        how='inner',
                    )
                    vals_i = merged[col_i].values
                    vals_j = merged[col_j].values
                else:
                    vals_i = df_i[col_i].dropna().values
                    vals_j = df_j[col_j].dropna().values

                min_len = min(len(vals_i), len(vals_j))
                if min_len < 2:
                    pval_matrix[i, j] = np.nan
                    continue

                vals_i = vals_i[:min_len]
                vals_j = vals_j[:min_len]

                try:
                    if pvalue_test == 'wilcoxon':
                        diff = vals_i - vals_j
                        if np.all(diff == 0):
                            pval_matrix[i, j] = 1.0
                        else:
                            _, p = wilcoxon(vals_i, vals_j)
                            pval_matrix[i, j] = p
                    elif pvalue_test == 'mannwhitneyu':
                        _, p = mannwhitneyu(vals_i, vals_j, alternative='two-sided')
                        pval_matrix[i, j] = p
                    else:
                        pval_matrix[i, j] = np.nan
                except Exception:
                    pval_matrix[i, j] = np.nan

        pval_df_out = pd.DataFrame(pval_matrix, index=col_names, columns=col_names)
        return pval_df_out

    return None