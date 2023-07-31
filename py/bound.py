import numpy as np
import scipy.integrate as integrate

# Initial scan parameters
MAX_LOG10EPS = 10.0
MIN_LOG10EPS = -10.0
NUM_EPS_SCAN = 10000
SCAN_GRID = np.logspace(MIN_LOG10EPS, MAX_LOG10EPS, NUM_EPS_SCAN)


def calculate_bounds(coh, coh_freqs, s, z):
    """
    Given a set of s, z, calculate the 95% likelihood using the jeffrey prior.
    The bound is evaluated for a particular frequency. As such, the primary loop

    is over different frequencies
    The steps For a given frequency are:
    1) Find for which eps the likelihood is maximum on a logspace grid in [MAX_LOG10EPS, MIN_LOG10EPS]
    2) Coordinate transformation from x = eps to y = logeps and apply rough renormalization

    s: (number of frequencies for this coherence chunk, number of coherence chunks, 3)
    z: (number of frequencies for this coherence chunk, number of coherence chunks, 3)
    """
    bounds = np.zeros(len(s))
    for (i, (sf, zf)) in enumerate(zip(s, z)):

        # 1) Find where eps is maximum in the logspace grid (with unknown norm set to 1)
        grid_logP = np.array(
            [calculate_logpdf(eps, sf, zf, norm=1.0) for eps in SCAN_GRID])
        max_idx = np.argmax(grid_logP)
        max_logP = grid_logP[max_idx]

        # 2) Transform pdf(x) to pdf(y)
        # for
        # x = eps
        # y = log(eps)
        # --> x(y) = exp(y)
        # --> dx(y)/dy = d/dy exp(y) = exp(y) ---> exp(logeps)
        # via
        # pdf(y) = pdf(x(y)) |dx(y)/dy|
        #        = pdf(exp(logeps)) * exp(logeps)
        #        = exp(logpdf(exp(logeps))) * exp(logeps)
        #        = exp(logpdf(exp(logeps)) + logeps)
        #
        # To add a rough normalization,
        #        = exp(logpdf(exp(logeps)) - lognorm + logeps)
        area_integrand = grid_logP - max_logP + np.log(SCAN_GRID)
        cumtrapz_area = integrate.cumulative_trapezoid(
            area_integrand, SCAN_GRID, initial=0)
        # Renormalize
        # In this line, we assume 100% of the mass is contained within [MAX_LOG10EPS, MIN_LOG10EPS]
        cumtrapz_area /= cumtrapz_area[-1]

        # Find index of first element which is >0.95
        idx_95 = find_cumsum_index(cumtrapz_area, 0.95)
        # eps = SCAN_GRID[idx_95]
        bounds[i] = SCAN_GRID[idx_95]

    np.savez(f"{coh}", f=coh_freqs, bounds=bounds)
    return coh_freqs, bounds

    # def transformed_unnormalized_pdf(logeps):
    #     eps = np.exp(logeps)
    #     logpdf_ = calculate_logpdf(1.0, eps)
    #     return np.exp(logpdf_ - max_logP + logeps)

    # logp_logeps_grid: Vec<f64> = log10eps_grid
    #     .iter()
    #     .map(|&log10eps| {
    #         eps = 10_f64.powf(log10eps);
    #         # see below for more info regarding coordinate transformation
    #         # log(pdf(logeps))
    #         #  = log(pdf(eps) * eps)
    #         #  = log(pdf(eps)) + log(eps)
    #         logpdf(1.0, eps) + eps.ln()
    #     }).collect();
    # tx.send((frequency, logp_logeps_grid.clone())).unwrap();
    # drop(tx);
    # (max_log10eps, max_logp) = log10eps_grid
    #     .into_iter()
    #     .zip(&logp_logeps_grid)
    #     .max_by(|a,b| a.1.partial_cmp(&b.1).unwrap()) # find max logpdf
    #     .unwrap();
    # max_logeps = max_log10eps / f64::exp(1.0).log10();
    # # log::debug!("found max_logp(logeps = {max_logeps:.3e}) = {max_logp:.2e}");
    # if max_logp.is_finite() {
    #     if max_logeps > 0.0 {
    #         return Err(BoundError::HighMax { max_logeps });
    #     }
    # } else {
    #     return Err(BoundError::InvalidMax);
    # }

    # 2) Transform pdf(x) to pdf(y)
    # for
    # x = eps
    # y = log(eps)
    # --> x(y) = exp(y)
    # --> dx(y)/dy = d/dy exp(y) = exp(y) ---> exp(logeps)
    # via
    # pdf(y) = pdf(x(y)) |dx(y)/dy|
    #        = pdf(exp(logeps)) * exp(logeps)
    #        = exp(logpdf(exp(logeps))) * exp(logeps)
    #        = exp(logpdf(exp(logeps)) + logeps)
    #
    # To add a rough normalization,
    #        = exp(logpdf(exp(logeps)) - lognorm + logeps)
    #
    # The integration library used below expects double precision

#     function = |upper: f64| {

#         # Try clenshaw curtis
#         mut result = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, 1e-10.ln(), upper, 1e-4).integral;

#         # Fallback if invalid
#         if !result.is_normal() {
#             mut num_steps = 1000;
#             while !result.is_normal() {
#                 hi_log10eps = -10.0;
#                 lo_log10eps = upper;
#                 dlog10eps = (lo_log10eps-hi_log10eps) / num_steps.sub(1) as f64;
#                 log10eps_grid = (0..num_steps).map(|i| hi_log10eps + i as f64 * dlog10eps);
#                 p_grid = log10eps_grid
#                     .map(|log10eps| {
#                         eps = 10_f64.powf(log10eps);
#                         logpdf(1.0, eps)
#                     });
#                 result = p_grid.tuple_windows().map(|(left_p, right_p)| {
#                     (left_p + right_p) * dlog10eps / 2.0
#                 }).sum();

#                 num_steps *= 10;
#             }
#             # One final, higher resolution integration
#             log::info!("failed clenshaw-curtis. using fallback value with {num_steps} steps");
#             hi_log10eps = -10.0;
#             lo_log10eps = upper;
#             dlogeps = (lo_log10eps-hi_log10eps) / num_steps.sub(1) as f64;
#             log10eps_grid = (0..num_steps).map(|i| hi_log10eps + i as f64 * dlogeps);
#             p_grid = log10eps_grid
#                 .map(|log10eps| {
#                     eps = 10_f64.powf(log10eps);
#                     logpdf(1.0, eps)
#                 });
#             result = p_grid.tuple_windows().map(|(left_p, right_p)| {
#                 (left_p + right_p) * dlogeps / 2.0
#             }).sum();
#         }

#         if !result.is_sign_positive() || !result.is_normal() {
#             return Err(BoundError::IntegralUnderflow)
#         }

#         Ok(result)
#     };

#     # Find 95% confidence interval
#     mut upper_bound_logeps = max_logeps;
#     #[allow(unused_mut)]
#     mut norm: f64 = quadrature::clenshaw_curtis::integrate(transformed_unnormalized_pdf, 1e-10.ln(), 1e1.ln(), 1e-4).integral;
#     if !norm.is_normal() {
#         return Err(BoundError::InvalidNorm { max_logeps })
#     }

#     initial_delta = 0.1;
#     tol = 1e-4;
#     target = 0.95;
#     max = 1000;
#     optimize(function, &mut upper_bound_logeps, initial_delta, target, norm, tol, max)?;

#     # Upper bound is for logeps so exponentiate to get exp bound
#     Ok(upper_bound_logeps.exp() as FloatType)
# }

# fn optimize<F: Fn(f64) -> std::result::Result<f64, BoundError>>(function: F, input: &mut f64, mut delta: f64, target: f64, norm: f64, tol: f64, max: u64) -> std::result::Result<u64, BoundError> {
#     const ABOVE: u8 = 1;
#     const BELOW: u8 = 2;
#     mut last_status = 0u8;
#     mut num_steps = 0;
#     loop {
#         current: f64 = function(*input)?;
#         assert!(current.is_finite(), "current is not finite f({input}) = {current}");
#         if current == 0.0 {
#             return Err(BoundError::IntegralUnderflow)
#         }
#         i = current / norm;
#         if (i-target).abs() < tol {
#             # If within tolerance we are converged
#             log::debug!("converged {i} vs {target} with input {input} (within tolerance {tol}) in {num_steps} steps (delta = {delta})");
#             break;
#         }

#         num_steps += 1;
#         if i > target {
#             # Else, if above target decrease upper_bound
#             # First check if we crossed target
#             if last_status == BELOW {
#                 # we have skipped over target, so reduce delta
#                 delta /= 10.0;

#             }
#             *input -= delta;
#             last_status = ABOVE;
#         } else if i < target {
#             # Else, if below target increase upper_bound
#             # First check if we crossed target
#             if last_status == ABOVE {
#                 # we have skipped over target, so reduce delta
#                 delta /= 10.0;

#             }
#             *input += delta;
#             last_status = BELOW;
#         }

#         log::trace!("step {num_steps}: input = {input}, {i} vs {target}, delta = {delta}");

#         if num_steps > max {
#             log::debug!("breaking with {} ({} < {}, delta = {})", input, (i-target).abs(), tol, delta);
#             return Err(BoundError::DidNotConverge)
#         }
#     }
#     Ok(num_steps)
# }


def calculate_logpdf(eps, sf, zf, norm=1.0):
    """
    Calculates logP at a given eps, 
    The pdf is of the form N * sqrt(sum(...)) * prod(a exp(b))
    so we will break down logpdf into summands:

    log( N * sqrt(sum(...)) * prod(a exp(b)) )
     = log(N) log(sqrt(sum(...)))  log(prod(a)) log(prod(exp(b)))
     = log(N) log(sqrt(sum(...)))  sum(log(a)) sum(log(exp(b)))
     = log(N) log(sqrt(sum(...)))  sum(log(a)) sum(b)

    So we identify the following summands:
    1) log N
    2) log sqrt(sum(...))
    3) sum(log(a))
    4) sum(b)

    Here
    - N is the normalization factor
    - (...) = 4 eps**2 s**4 / (3 + eps**2 s**2)**2
    - a = 1 / (3 eps**2 s**2)
    - b = -3 z**2 / (3 + eps**2 s**2)
    """

    # Term 1: log N
    term_1 = np.log(norm)

    # Term 2: log sqrt(sum(...))
    # (this is from jeffery's prior)
    term_2 = np.log(
        np.sqrt(np.sum(4 * eps**2 * sf**4 / (3 + eps**2 * sf**2)**2)))

    # Term 3: sum(log(a))
    term_3 = np.sum(np.log(1 / (3 * eps**2 * sf**2)))

    # Term 4: sum(b)
    term_4 = np.sum(-3 * np.abs(zf)**2 / (3 + eps**2 * sf**2))

    return term_1 + term_2 + term_3 + term_4


def find_cumsum_index(arr, target):
    cumsum = np.cumsum(arr)
    idx = np.searchsorted(cumsum, target)
    if idx == len(arr):
        return -1
    return idx
