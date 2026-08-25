\# Sensitivity Analysis



\## 1. Purpose



Sensitivity analysis evaluates how changes in uncertain physical parameters affect the predictions of the AVANT analytical strain model and its agreement with the canonical 1-Darcy reference dataset.



The production sensitivity runner is



```text

scripts/sensitivity/run\_sensitivity\_1darcy.py

```



and can be inspected with



```bash

python scripts/sensitivity/run\_sensitivity\_1darcy.py --help

```



The sensitivity workflow complements Bayesian inversion and optimal experimental design (OED), but these analyses answer different questions.



\- \*\*Sensitivity analysis:\*\* How strongly does the model response change when a parameter changes?

\- \*\*Bayesian inversion:\*\* Which parameter values are supported by the observed data, including posterior uncertainty and parameter correlation?

\- \*\*OED:\*\* Which observations or sensor configurations provide the most useful information about the uncertain parameters?



A parameter may strongly influence model output while still being difficult to identify uniquely if its effect is correlated with another parameter.



\---



\## 2. Production Parameter Set



The current sensitivity analysis uses the same six physical parameters as the production Bayesian inversion:



\\\[

\\theta\_{\\mathrm{physical}}

=

\[

a,\\,

b,\\,

h,\\,

\\theta\_{\\mathrm{deg}},\\,

x\_0',\\,

y\_0'

].

\\]



Their current reference values are:



| Parameter | Reference value |

|---|---:|

| \\(a\\) | 190 m |

| \\(b\\) | 325 m |

| \\(h\\) | 520 m |

| \\(\\theta\_{\\mathrm{deg}}\\) | 15° |

| \\(x\_0'\\) | 45 m |

| \\(y\_0'\\) | 170 m |



The current analysis bounds are:



| Parameter | Lower | Upper |

|---|---:|---:|

| \\(a\\) | 40 m | 900 m |

| \\(b\\) | 20 m | 1000 m |

| \\(h\\) | 50 m | 900 m |

| \\(\\theta\_{\\mathrm{deg}}\\) | -90° | 90° |

| \\(x\_0'\\) | -900 m | 900 m |

| \\(y\_0'\\) | -900 m | 900 m |



These bounds are aligned with the physical parameter domain used by the current Bayesian inversion and OED analyses.



\---



\## 3. Fixed Analytical Inputs



The current 1-Darcy sensitivity workflow holds the following analytical-model quantities fixed:



| Quantity | Value |

|---|---:|

| \\(c\\) | 3.125 m |

| \\(E\\) | \\(8.0\\times10^9\\) Pa |

| \\(\\nu\\) | 0.35 |

| \\(p\_{\\max}\\) | 999310 Pa |

| \\(t\_{\\mathrm{peak}}\\) | 350000 s |

| \\(d\\) | 0.4 |

| \\(\\alpha\\) | 0.8 |



Therefore the reported sensitivity results are conditional on these fixed values.



Sensitivity to these quantities is not represented unless they are explicitly added to the sensitivity parameter set.



\---



\## 4. Canonical Data Contract



The production sensitivity analysis uses the same canonical dataset as the Bayesian inversion:



```text

datasets/comsol/1darcy/metadata/stations.csv

datasets/comsol/1darcy/processed/strain.csv

```



The data contain:



```text

8 stations

107 time steps

4 strain components per station

32 strain channels

3424 strain observations

```



The observed components are



\\\[

\\epsilon\_{xx},\\qquad

\\epsilon\_{yy},\\qquad

\\epsilon\_{zz},\\qquad

\\epsilon\_{xy}.

\\]



Using the same data contract across inversion, sensitivity analysis, and OED prevents the different analyses from silently operating on incompatible station networks or strain layouts.



\---



\## 5. Baseline State



Sensitivity calculations begin from a reference physical parameter state:



\\\[

\[

a,b,h,\\theta,x\_0',y\_0'

]

=

\[

190,\\,

325,\\,

520,\\,

15,\\,

45,\\,

170

].

\\]



The analytical prediction at this reference state is compared with the canonical observed strain dataset.



A global strain error metric is used to characterize baseline model-data mismatch.



For observed values \\(y\_i\\) and model predictions \\(\\hat y\_i\\),



\\\[

\\mathrm{RMSE}

=

\\sqrt{

\\frac{1}{N}

\\sum\_{i=1}^{N}

(y\_i-\\hat y\_i)^2

}.

\\]



This baseline provides a common reference against which parameter perturbations can be compared.



\---



\## 6. Complementary Sensitivity Methods



The production workflow combines several methods rather than relying on a single sensitivity metric.



The main analyses are:



1\. one-at-a-time parameter sweeps;

2\. two-dimensional response surfaces;

3\. global parameter sampling; and

4\. Sobol variance-based sensitivity analysis.



Each method provides different information about the model.



\---



\## 7. One-at-a-Time Sensitivity



One-at-a-time (OAT) sensitivity varies one parameter while holding all remaining parameters at their reference values.



For parameter \\(\\theta\_j\\),



\\\[

\\theta\_j

\\in

\[\\theta\_{j,\\min},\\theta\_{j,\\max}]

\\]



while



\\\[

\\theta\_{k\\ne j}

=

\\theta\_{k,\\mathrm{reference}}.

\\]



This produces an interpretable response curve for each parameter.



OAT analysis can reveal:



\- response magnitude;

\- monotonic or non-monotonic behavior;

\- approximate local curvature;

\- regions of weak sensitivity;

\- parameter values associated with lower model-data RMSE.



\### Limitation



OAT analysis does not represent interactions among parameters.



A parameter that appears weak when varied alone may become important through interaction with another parameter.



OAT should therefore be interpreted together with the multidimensional analyses.



\---



\## 8. Two-Dimensional Response Surfaces



Two-dimensional response surfaces vary pairs of parameters simultaneously.



For parameters \\(\\theta\_i\\) and \\(\\theta\_j\\),



\\\[

R(\\theta\_i,\\theta\_j)

\\]



is evaluated over a grid while the other parameters remain fixed.



These surfaces are useful for identifying:



\- parameter tradeoffs;

\- correlated response directions;

\- curved low-error regions;

\- multimodal structure;

\- non-unique parameter combinations;

\- strong pairwise interactions.



For example, inclusion dimensions and depth may partially compensate for each other in their effect on observed strain.



Similarly, orientation and horizontal center location may interact because both alter the spatial strain pattern observed by the station network.



\---



\## 9. Why Response Surfaces Matter for Inversion



A low-RMSE valley in a two-dimensional response surface often indicates that multiple parameter combinations can produce similar predictions.



Such behavior can later appear in Bayesian inference as:



\- posterior correlation;

\- broad marginal distributions;

\- elongated joint posterior structure;

\- multimodality;

\- slow MCMC convergence.



Therefore response surfaces provide useful diagnostic context for interpreting posterior behavior.



They are not, however, a replacement for Bayesian inference because they do not directly represent posterior probability.



\---



\## 10. Global Parameter Sampling



Global sensitivity analysis samples the multidimensional parameter domain rather than perturbing only one or two parameters around a reference state.



For each sampled state,



\\\[

\\theta^{(k)}

=

\[

a,b,h,\\theta,x\_0',y\_0'

]^{(k)},

\\]



the analytical forward model is evaluated and one or more scalar response measures are calculated.



Global sampling can identify:



\- parameter combinations with low model-data mismatch;

\- broad nonlinear trends;

\- parameter interactions;

\- regions where the analytical model becomes poorly conditioned;

\- global behavior that is invisible in local OAT sweeps.



The production runner provides a configurable number of global samples.



\---



\## 11. Sobol Sensitivity Analysis



Sobol analysis is a variance-based global sensitivity method.



For model response \\(Y=f(\\theta)\\), the total variance can be decomposed into contributions associated with individual parameters and their interactions.



\### First-order index



The first-order Sobol index for parameter \\(i\\),



\\\[

S\_i,

\\]



measures the fraction of output variance explained by that parameter alone.



Conceptually,



\\\[

S\_i

=

\\frac{

\\mathrm{Var}\_{\\theta\_i}

\\left\[

\\mathbb{E}(Y\\mid\\theta\_i)

\\right]

}{

\\mathrm{Var}(Y)

}.

\\]



\### Total-order index



The total-order index



\\\[

S\_{T\_i}

\\]



measures the contribution of parameter \\(i\\), including all interactions involving that parameter.



A large difference between



\\\[

S\_{T\_i}

\\]



and



\\\[

S\_i

\\]



suggests that interactions are important.



\---



\## 12. Sobol Base Sample Size



The production runner exposes the Sobol base sample size through



```text

\--sobol-base

```



The total number of required forward-model evaluations is larger than the base sample because the Sobol/Saltelli design constructs additional parameter combinations.



Small values such as



```text

\--sobol-base 32

```



are useful for software verification and preliminary exploration.



They should not automatically be treated as publication-quality Sobol estimates.



Production Sobol analysis should use a sufficiently large base sample and should verify stability of the resulting sensitivity indices.



\---



\## 13. OAT Resolution



The number of points used in one-at-a-time sweeps is controlled by



```text

\--oat-points

```



For example,



```text

\--oat-points 7

```



provides a lightweight diagnostic sweep.



Larger values provide smoother response curves and better resolution of nonlinear behavior at increased computational cost.



\---



\## 14. Response-Surface Resolution



The two-dimensional surface resolution is controlled by



```text

\--heatmap-points

```



A value such as



```text

\--heatmap-points 9

```



produces a \\(9\\times9\\) grid for each selected parameter pair.



Because the number of model evaluations grows quadratically with grid resolution, dense surfaces are more expensive than one-dimensional sweeps.



Production settings should balance smoothness, computational cost, and the number of parameter pairs being evaluated.



\---



\## 15. Global Sample Count



The number of globally sampled parameter states is controlled by



```text

\--global-samples

```



For example,



```text

\--global-samples 100

```



is appropriate for a preliminary software or exploratory run.



Larger global ensembles provide better coverage of the six-dimensional parameter space.



\---



\## 16. Smoke-Test Example



A lightweight sensitivity run can be launched with:



```bash

python scripts/sensitivity/run\_sensitivity\_1darcy.py \\

&#x20;   --sobol-base 32 \\

&#x20;   --oat-points 7 \\

&#x20;   --heatmap-points 9 \\

&#x20;   --global-samples 100

```



This configuration is useful for:



\- verifying the complete sensitivity workflow;

\- checking output generation;

\- detecting runtime errors;

\- estimating computational cost;

\- previewing figure layouts.



It should not automatically be interpreted as the final production sensitivity analysis.



\---



\## 17. Production Sensitivity Runs



Production settings should use higher resolution than software smoke tests.



The appropriate values depend on:



\- available computational resources;

\- forward-model runtime;

\- desired response-surface resolution;

\- required stability of Sobol indices;

\- number of parameter pairs analyzed.



For final analyses, convergence/stability of sensitivity metrics should be checked by increasing sample counts and confirming that major conclusions do not change materially.



Large sensitivity analyses are intended for HPC execution.



See



```text

docs/palmetto.md

```



for the production deployment strategy.



\---



\## 18. Sensitivity Versus Identifiability



Sensitivity and identifiability are related but not equivalent.



\### Sensitivity



A parameter is sensitive if changing it causes a substantial change in the model response.



\### Identifiability



A parameter is identifiable if the observations contain enough independent information to estimate it distinctly from the other parameters.



A highly sensitive parameter can still be poorly identifiable if another parameter produces a similar model response.



This is why the project combines sensitivity analysis with:



\- Bayesian posterior analysis;

\- Fisher-information diagnostics;

\- station-subset OED;

\- robust OED;

\- posterior-informed Bayesian OED.



\---



\## 19. Sensitivity Versus Bayesian Posterior Width



A narrow posterior does not necessarily imply that a parameter has the largest global sensitivity index.



Posterior width depends on:



\- the observations;

\- parameter correlations;

\- prior bounds;

\- residual uncertainty;

\- model structure;

\- sensitivity around parameter combinations supported by the data.



Sobol sensitivity instead describes variance over a specified parameter domain.



The two analyses therefore answer different scientific questions.



\---



\## 20. Sensitivity Versus OED



Sensitivity asks:



> Which parameters strongly influence the model response?



OED asks:



> Which observations or sensor configurations provide the most useful independent information about the uncertain parameters?



A location with a large strain amplitude is not automatically the optimal sensor location.



For example, several nearby sensors may all respond strongly but provide redundant information.



OED accounts for the multidimensional parameter-information structure rather than only response magnitude.



\---



\## 21. Interpreting Center-Location Sensitivity



The parameters



\\\[

x\_0',\\qquad y\_0'

\\]



control the horizontal position of the inclusion relative to the monitoring array.



Their sensitivity is inherently spatial.



Changes in center location alter:



\- station-to-inclusion distances;

\- relative azimuths;

\- tensor-component patterns;

\- which stations experience the strongest response.



This is one reason that multi-station measurements are particularly valuable for estimating inclusion location.



\---



\## 22. Interpreting Orientation Sensitivity



The parameter



\\\[

\\theta\_{\\mathrm{deg}}

\\]



rotates the inclusion relative to the observation coordinate system.



Orientation can strongly affect the horizontal strain tensor and spatial pattern.



The implementation uses a defined horizontal tensor-rotation convention that is protected by regression tests. These verify:



\- zero rotation leaves all strain components unchanged;

\- horizontal rotation leaves the vertical normal strain \\(\\epsilon\_{zz}\\) unchanged;

\- the in-plane trace \\(\\epsilon\_{xx}+\\epsilon\_{yy}\\) is invariant under rotation;

\- a 90° rotation swaps \\(\\epsilon\_{xx}\\) and \\(\\epsilon\_{yy}\\) and gives the shear sign required by the adopted coordinate convention; and

\- applying a rotation by \\(\\theta\\) followed by \\(-\\theta\\) recovers the original strain tensor.





\## 23. Interpreting Size and Depth Sensitivity



The parameters



\\\[

a,\\qquad b,\\qquad h

\\]



control inclusion dimensions and depth.



Their effects can be correlated because each influences the magnitude and spatial decay of the strain field.



For example, changes in inclusion size may partially compensate for changes in depth.



Such interactions should be examined using:



\- pairwise response surfaces;

\- global sampling;

\- Bayesian posterior correlations;

\- Fisher-information diagnostics.



\---



\## 24. Model-Data Mismatch



Sensitivity analyses that use RMSE or related error measures evaluate not only model response magnitude but also agreement with the reference dataset.



A lower RMSE indicates closer agreement under the chosen scalar metric.



However, a global RMSE can hide structured mismatch.



Therefore final interpretation should also examine:



\- individual stations;

\- individual tensor components;

\- temporal residual patterns;

\- posterior predictive diagnostics.



The Bayesian postprocessor provides these more detailed diagnostics.



\---



\## 25. Output Philosophy



Sensitivity outputs are generated analysis products and are not part of the version-controlled source repository.



They should be written under the configured results/output hierarchy.



Publication-quality figures can be regenerated from the production scripts and archived with the associated analysis run.



Curated figures intended for repository documentation may be copied intentionally to



```text

assets/

```



\---



\## 26. Reproducibility



Every production sensitivity run should preserve:



```text

Git commit

dataset

station metadata

parameter bounds

reference parameter state

fixed analytical inputs

Sobol base size

OAT resolution

response-surface resolution

global sample count

random seed

software environment

execution command

output directory

```



This information should accompany final sensitivity figures and tables.



\---



\## 27. Recommended Interpretation Sequence



A useful interpretation order is:



```text

1\. Baseline model-data agreement

&#x20;            |

&#x20;            v

2\. OAT response curves

&#x20;            |

&#x20;            v

3\. Pairwise response surfaces

&#x20;            |

&#x20;            v

4\. Global parameter sampling

&#x20;            |

&#x20;            v

5\. Sobol indices

&#x20;            |

&#x20;            v

6\. Compare with Bayesian posterior

&#x20;            |

&#x20;            v

7\. Compare with Fisher/OED results

```



Agreement among these analyses strengthens interpretation.



Disagreement is also scientifically useful because it can reveal parameter interactions, observation limitations, or prior dependence.



\---



\## 28. Related Documentation



See:



```text

docs/workflow.md

docs/bayesian\_inversion.md

docs/optimal\_experimental\_design.md

docs/reproducibility.md

docs/palmetto.md

```



for the complete workflow, Bayesian inference, experimental design, reproducibility, and HPC execution.

