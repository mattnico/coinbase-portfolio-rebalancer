"""
Monte Carlo visualization tools for equity curve analysis.

Generates interactive charts showing Monte Carlo simulation paths
alongside original historical performance.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import List, TYPE_CHECKING
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

if TYPE_CHECKING:
    from src.monte_carlo_simulator import SimulationResult

logger = logging.getLogger(__name__)


def create_equity_curve_chart(
    mc_results: List['SimulationResult'],
    original_result: 'SimulationResult',
    strategy_name: str
) -> 'go.Figure':
    """
    Create interactive equity curve chart showing Monte Carlo percentile bands and original.

    Args:
        mc_results: List of Monte Carlo simulation results
        original_result: Result from original (unshuffled) historical data
        strategy_name: Name of the strategy for chart title

    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly")

    fig = go.Figure()

    # Determine global start time from both MC results and original
    # This ensures consistent x-axis alignment
    all_start_times = []

    # Collect start times from MC results
    if mc_results:
        for result in mc_results:
            timestamps = [state.timestamp for state in result.portfolio_history]
            if timestamps:
                all_start_times.append(timestamps[0])

    # Collect start time from original result
    orig_timestamps = [state.timestamp for state in original_result.portfolio_history]
    if orig_timestamps:
        all_start_times.append(orig_timestamps[0])

    # Use the earliest start time across ALL simulations (MC + original)
    if not all_start_times:
        return fig

    global_start_time = min(all_start_times)
    logger.info(f"Using global start time for visualization: {global_start_time}")

    # Calculate percentile bands from all Monte Carlo simulations
    if mc_results:
        # Find the full date range across all simulations
        all_timestamps = set()

        for result in mc_results:
            timestamps = [state.timestamp for state in result.portfolio_history]
            if timestamps:
                all_timestamps.update(timestamps)

        if not all_timestamps:
            return fig

        # Convert to sorted list of unique days (using global start time)
        unique_timestamps = sorted(all_timestamps)
        days = [(ts - global_start_time).days for ts in unique_timestamps]

        # Build matrix: rows = simulations, columns = time points
        # For each simulation, map its values to the common timeline
        values_matrix = []

        for result in mc_results:
            # Create a lookup from timestamp to value
            timestamp_to_value = {
                state.timestamp: state.total_value_usd
                for state in result.portfolio_history
            }

            # For each timestamp in the common timeline, get the value
            # If exact timestamp doesn't exist, use the most recent prior value
            sim_values = []
            last_value = None

            for ts in unique_timestamps:
                if ts in timestamp_to_value:
                    last_value = timestamp_to_value[ts]
                    sim_values.append(last_value)
                elif last_value is not None:
                    # Use last known value (forward fill)
                    sim_values.append(last_value)
                else:
                    # No prior value yet, this shouldn't happen if data is properly ordered
                    sim_values.append(np.nan)

            values_matrix.append(sim_values)

        values_matrix = np.array(values_matrix)  # Shape: (num_sims, num_timepoints)

        # Calculate percentiles at each time point (ignoring NaN values)
        p5 = np.nanpercentile(values_matrix, 5, axis=0)
        p25 = np.nanpercentile(values_matrix, 25, axis=0)
        p50 = np.nanpercentile(values_matrix, 50, axis=0)  # median
        p75 = np.nanpercentile(values_matrix, 75, axis=0)
        p95 = np.nanpercentile(values_matrix, 95, axis=0)

        # Add 5th-95th percentile band (lightest blue)
        fig.add_trace(go.Scatter(
            x=days,
            y=p95,
            mode='lines',
            name='5th-95th Percentile',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=p5,
            mode='lines',
            name='5th-95th Percentile',
            line=dict(width=0),
            fillcolor='rgba(100, 150, 255, 0.2)',
            fill='tonexty',
            showlegend=True,
            hovertemplate='Day %{x}<br>5th: $%{y:,.2f}<extra></extra>'
        ))

        # Add 25th-75th percentile band (darker blue)
        fig.add_trace(go.Scatter(
            x=days,
            y=p75,
            mode='lines',
            name='25th-75th Percentile (IQR)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=days,
            y=p25,
            mode='lines',
            name='25th-75th Percentile (IQR)',
            line=dict(width=0),
            fillcolor='rgba(100, 150, 255, 0.4)',
            fill='tonexty',
            showlegend=True,
            hovertemplate='Day %{x}<br>25th: $%{y:,.2f}<extra></extra>'
        ))

        # Add median line (dashed blue)
        fig.add_trace(go.Scatter(
            x=days,
            y=p50,
            mode='lines',
            name='Median (50th Percentile)',
            line=dict(color='rgb(50, 100, 200)', width=2, dash='dash'),
            hovertemplate='Day %{x}<br>Median: $%{y:,.2f}<extra></extra>'
        ))

    # Add original strategy line (dark green, thick)
    # Note: orig_timestamps was already extracted earlier for global_start_time calculation
    orig_values = [state.total_value_usd for state in original_result.portfolio_history]

    if orig_timestamps:
        # Use the same global start time for consistency
        orig_days = [(ts - global_start_time).days for ts in orig_timestamps]

        fig.add_trace(go.Scatter(
            x=orig_days,
            y=orig_values,
            mode='lines',
            name='Original Strategy',
            line=dict(color='rgb(0, 128, 0)', width=3),
            hovertemplate='Day %{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))

    # Update layout
    num_sims = len(mc_results)
    initial_value = original_result.initial_value
    final_value_orig = original_result.final_value
    return_orig = original_result.total_return_percent

    fig.update_layout(
        title=dict(
            text=f'Monte Carlo Equity Curves - {strategy_name}<br>' +
                 f'<sub>{num_sims} simulations | Original: ${initial_value:,.0f} → ${final_value_orig:,.0f} ({return_orig:.1f}%)</sub>',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Days',
            gridcolor='lightgray',
            showgrid=True
        ),
        yaxis=dict(
            title='Portfolio Value ($)',
            gridcolor='lightgray',
            showgrid=True,
            tickformat='$,.0f'
        ),
        plot_bgcolor='white',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        width=1200,
        height=700
    )

    return fig


def save_monte_carlo_chart(
    fig: 'go.Figure',
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    num_simulations: int,
    chunk_days: int,
    seed: int,
    output_dir: str = "results"
) -> Path:
    """
    Save Monte Carlo chart to HTML file with descriptive filename.

    Args:
        fig: Plotly figure to save
        strategy_name: Strategy name (e.g., "Adaptive", "Top_Three")
        start_date: Simulation start date
        end_date: Simulation end date
        num_simulations: Number of Monte Carlo simulations
        chunk_days: Chunk size used in shuffling
        seed: Random seed used for reproducibility
        output_dir: Directory to save file (default: "results")

    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Format filename with all relevant variables
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Sanitize strategy name for filename
    safe_strategy = strategy_name.replace(" ", "_").replace("(", "").replace(")", "")

    filename = f"monte_carlo_{safe_strategy}_{start_str}_{end_str}_{num_simulations}sims_{chunk_days}d_chunks_seed{seed}.html"
    filepath = output_path / filename

    # Save as interactive HTML
    fig.write_html(
        str(filepath),
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )

    logger.info(f"📊 Saved Monte Carlo equity curve: {filepath}")
    return filepath


def generate_and_save_chart(
    mc_results: List['SimulationResult'],
    original_result: 'SimulationResult',
    strategy_name: str,
    start_date: datetime,
    end_date: datetime,
    num_simulations: int,
    chunk_days: int,
    seed: int,
    output_dir: str = "results"
) -> Path:
    """
    Convenience function to create and save chart in one call.

    Args:
        mc_results: List of Monte Carlo simulation results
        original_result: Result from original historical data
        strategy_name: Strategy name
        start_date: Simulation start date
        end_date: Simulation end date
        num_simulations: Number of simulations
        chunk_days: Chunk size
        seed: Random seed used for reproducibility
        output_dir: Output directory

    Returns:
        Path to saved chart file
    """
    try:
        fig = create_equity_curve_chart(mc_results, original_result, strategy_name)
        filepath = save_monte_carlo_chart(
            fig, strategy_name, start_date, end_date,
            num_simulations, chunk_days, seed, output_dir
        )
        return filepath
    except ImportError as e:
        logger.warning(f"Cannot generate chart: {e}")
        logger.warning("Install plotly to enable visualizations: pip install plotly")
        return None
    except Exception as e:
        logger.error(f"Error generating chart: {e}", exc_info=True)
        return None
