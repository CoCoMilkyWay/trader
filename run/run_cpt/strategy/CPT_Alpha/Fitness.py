"""
rather than using IC information coeffcient as a measure of how correlated 2 things are, in financial alpha mining, what if I waant to use 1 sided correlation? e.g. if a goes higher, then b goes higher, but not necessarily the other around?
EditThis is a great question about asymmetric relationships in financial signals. Let me suggest a few approaches that could help capture one-sided correlations:

Semi-correlation (Upside/Downside correlation)


Calculate correlation only when the predictor variable (A) is increasing
This directly measures what you're looking for: correlation conditional on A going up
You can implement this by filtering your data to only include periods where ΔA > 0, then calculate regular correlation


Copula-based measures


Tail dependence coefficients can measure asymmetric relationships
Upper tail dependence specifically looks at how often B is extreme when A is extreme
This could be more appropriate than semi-correlation if you care about magnitude
"""