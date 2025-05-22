Here we propose two new imputation techniques: Controlled Weighted Rational Bezier Curves (CRBC) and Controlled PCHIP with Mapped Peak and Valleys
of Control Points (CMPV). Both methods begin by linearly imputing five values at the start and end of each missing segment to ensure smooth transitions. 
CRBC employs Rational Bézier-Bernstein curves, using up to 900 control points from surrounding HR data, with higher weights assigned to peaks and valleys to
preserve physiological dynamics.
In contrast, CMPV uses PCHIP interpolation, where peaks and valleys from the preceding and succeeding control points are inverted and symmetrically mapped into the imputation region,
allowing the model to reflect natural HR fluctuations. Both methods constrain the final imputed values within the clinically realistic range of 40–160 bpm. Our proposed imputation 
methods are evaluated on HR values of the D1NAMO dataset against state-of-the-art techniques—Linear, KNN, PCHIP, and B-Spline—using a combination of standard and data and pattern
centric metrics. Performance is assessed through RMSE alongside our proposed metrics: EDM and PAS, which capture signal pattern fidelity. Finally, a multidimensional combined score, 
derived from a weighted algebraic sum of all three metrics, provides a comprehensive evaluation of imputation quality.


