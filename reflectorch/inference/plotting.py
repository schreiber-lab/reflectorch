from matplotlib import pyplot as plt
import numpy as np


def plot_prediction_results(
    prediction_dict: dict,
    q_exp: np.ndarray = None,
    curve_exp: np.ndarray = None,
    sigmas_exp: np.ndarray = None,
    q_model: np.ndarray = None,
):
    """
    Plot the experimental curve (with optional error bars), the predicted
    and polished curves, and also the predicted/polished SLD profiles.

    Args:
        prediction_dict (dict): Dictionary containing 'predicted_curve',
                                'predicted_sld_profile', 'predicted_sld_xaxis',
                                and optionally 'polished_curve', 'sld_profile_polished'.
        q_exp (ndarray, optional): Experimental q-values.
        curve_exp (ndarray, optional): Experimental reflectivity curve.
        sigmas_exp (ndarray, optional): Error bars of the experimental reflectivity.
        q_model (ndarray, optional): The q-values on which prediction_dict's reflectivity
                                     was computed (e.g. from EasyInferenceModel.interpolate_data_to_model_q).

    Example usage:
        prediction_dict = model.predict(...)
        plot_prediction_results(
            prediction_dict,
            q_exp=q_exp,
            curve_exp=curve_exp,
            sigmas_exp=sigmas_exp,
            q_model=q_model
        )
    """

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # --- Left plot: Reflectivity curves ---
    ax[0].set_yscale('log')
    ax[0].set_xlabel('q [$Å^{-1}$]', fontsize=20)
    ax[0].set_ylabel('R(q)', fontsize=20)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].tick_params(axis='both', which='minor', labelsize=15)

    # Optionally set major y ticks (log scale)
    y_tick_locations = [10 ** (-2 * i) for i in range(6)]
    ax[0].yaxis.set_major_locator(plt.FixedLocator(y_tick_locations))

    # Plot experimental data with error bars (if provided)
    if q_exp is not None and curve_exp is not None:
        el = ax[0].errorbar(
            q_exp, curve_exp, yerr=sigmas_exp,
            xerr=None, c='b', ecolor='purple', elinewidth=1,
            marker='o', linestyle='none', markersize=3,
            label='exp. curve', zorder=1
        )
        # Change the color of error bar lines (optional)
        elines = el.get_children()
        if len(elines) > 1:
            elines[1].set_color('purple')

    # Plot predicted curve
    if 'predicted_curve' in prediction_dict and q_model is not None:
        ax[0].plot(q_model, prediction_dict['predicted_curve'], c='red', lw=2, label='pred. curve')

    # Plot polished curve (if present)
    if 'polished_curve' in prediction_dict and q_model is not None:
        ax[0].plot(q_model, prediction_dict['polished_curve'], c='orange', ls='--', lw=2, label='polished pred. curve')

    ax[0].legend(fontsize=12)

    # --- Right plot: SLD profiles ---
    ax[1].set_xlabel('z [$Å$]', fontsize=20)
    ax[1].set_ylabel('SLD [$10^{-6} Å^{-2}$]', fontsize=20)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    ax[1].tick_params(axis='both', which='minor', labelsize=15)

    # Predicted SLD
    if 'predicted_sld_xaxis' in prediction_dict and 'predicted_sld_profile' in prediction_dict:
        ax[1].plot(
            prediction_dict['predicted_sld_xaxis'],
            prediction_dict['predicted_sld_profile'],
            c='red', label='pred. sld'
        )

    # Polished SLD
    if 'sld_profile_polished' in prediction_dict and 'predicted_sld_xaxis' in prediction_dict:
        ax[1].plot(
            prediction_dict['predicted_sld_xaxis'],
            prediction_dict['sld_profile_polished'],
            c='orange', ls='--', label='polished sld'
        )

    ax[1].legend(fontsize=12)

    plt.tight_layout()
    plt.show()