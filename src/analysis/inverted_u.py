"""Inverted-U analysis: test whether humor quality peaks at moderate
values of distributional metrics (surprisal, entropy, embedding distance)."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass
class InvertedUResult:
    """Results from fitting a quadratic model: y = a*x^2 + b*x + c."""

    metric_name: str
    humor_dimension: str
    linear_coeff: float  # b
    quadratic_coeff: float  # a (negative = inverted-U)
    intercept: float  # c
    r_squared: float
    p_value_quadratic: float  # significance of the quadratic term
    n_samples: int
    peak_x: float | None  # x value at the peak (-b / 2a), if inverted-U
    is_inverted_u: bool  # quadratic_coeff < 0 and significant


def fit_inverted_u(
    metric_values: np.ndarray,
    humor_scores: np.ndarray,
    metric_name: str = "",
    humor_dimension: str = "overall",
    alpha: float = 0.05,
) -> InvertedUResult:
    """
    Fit y = a*x^2 + b*x + c and test for significant inverted-U.

    A significant negative quadratic coefficient supports the inverted-U
    hypothesis: humor peaks at moderate metric values.
    """
    mask = np.isfinite(metric_values) & np.isfinite(humor_scores)
    x = metric_values[mask]
    y = humor_scores[mask]
    n = len(x)

    if n < 10:
        return InvertedUResult(
            metric_name=metric_name,
            humor_dimension=humor_dimension,
            linear_coeff=0.0,
            quadratic_coeff=0.0,
            intercept=0.0,
            r_squared=0.0,
            p_value_quadratic=1.0,
            n_samples=n,
            peak_x=None,
            is_inverted_u=False,
        )

    # Standardize x for numerical stability
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, 1)).ravel()

    # Build design matrix [x^2, x, 1]
    X = np.column_stack([x_scaled**2, x_scaled, np.ones(n)])

    # OLS fit
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coeffs

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # F-test for quadratic term: compare full model vs linear-only
    X_linear = np.column_stack([x_scaled, np.ones(n)])
    coeffs_lin, *_ = np.linalg.lstsq(X_linear, y, rcond=None)
    y_pred_lin = X_linear @ coeffs_lin
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)

    if ss_res > 0 and n > 3:
        f_stat = ((ss_res_lin - ss_res) / 1) / (ss_res / (n - 3))
        p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
    else:
        p_value = 1.0

    # Peak location in original scale
    peak_x = None
    if a < 0 and abs(a) > 1e-10:
        peak_scaled = -b / (2 * a)
        peak_x = float(
            scaler.inverse_transform(np.array([[peak_scaled]]))[0, 0]
        )

    return InvertedUResult(
        metric_name=metric_name,
        humor_dimension=humor_dimension,
        linear_coeff=float(b),
        quadratic_coeff=float(a),
        intercept=float(c),
        r_squared=float(r_squared),
        p_value_quadratic=float(p_value),
        n_samples=n,
        peak_x=peak_x,
        is_inverted_u=(a < 0 and p_value < alpha),
    )


def run_inverted_u_analysis(
    metrics_dict: dict[str, np.ndarray],
    humor_dict: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> list[InvertedUResult]:
    """
    Test inverted-U for all metric × humor-dimension combinations.

    Args:
        metrics_dict: {metric_name: array of values}
        humor_dict: {humor_dimension: array of scores}

    Returns:
        List of InvertedUResult for each combination.
    """
    results = []
    for metric_name, metric_vals in metrics_dict.items():
        for humor_dim, humor_vals in humor_dict.items():
            result = fit_inverted_u(
                metric_vals,
                humor_vals,
                metric_name=metric_name,
                humor_dimension=humor_dim,
                alpha=alpha,
            )
            results.append(result)
    return results


def save_results(results: list[InvertedUResult], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Saved {len(results)} inverted-U results to {output_path}")
