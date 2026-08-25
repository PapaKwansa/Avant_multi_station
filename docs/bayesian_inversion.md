\# Bayesian Inversion



\## 1. Purpose



The AVANT Bayesian inversion estimates the geometry, orientation, depth, and horizontal location of the pressurized subsurface inclusion from multi-station transient strain observations.



The production implementation separates:



1\. \*\*physical parameters of interest\*\*, which describe the inclusion;

2\. \*\*fixed analytical-model inputs\*\*, which are not currently inferred; and

3\. \*\*a statistical nuisance parameter\*\*, which represents the effective residual scale between the analytical model and the observed/reference strain data.



The reusable Bayesian likelihood is implemented in



```text

src/avant\_model/inversion/bayesian\_inversion\_multi\_station.py

```



and the production PyDREAM runner is



```text

scripts/inversion/run\_inversion\_multi\_station.py

```



Posterior analysis is performed with



```text

scripts/postprocessing/postprocess\_bayesian\_1darcy.py

```



\---



\## 2. Canonical Observations



The production inversion uses



```text

datasets/comsol/1darcy/metadata/stations.csv

datasets/comsol/1darcy/processed/strain.csv

```



with the following contract:



```text

stations              8

station names         S01 ... S08

time steps            107

strain components     4

strain channels       32

total observations    3424

```



The observed components are



\\\[

\\epsilon\_{xx},\\qquad

\\epsilon\_{yy},\\qquad

\\epsilon\_{zz},\\qquad

\\epsilon\_{xy}.

\\]



The likelihood requires the observed and predicted vectors to use identical ordering.



The canonical component-major ordering is



```text

eXX for all stations

eYY for all stations

eXY for all stations

eZZ for all stations

```



with the complete time series for each selected channel concatenated into the likelihood vector.



\---



\## 3. Physical Parameters



The six inferred physical parameters are



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



| Parameter | Units | Interpretation |

|---|---:|---|

| \\(a\\) | m | inclusion semi-dimension along the local x direction |

| \\(b\\) | m | inclusion semi-dimension along the local y direction |

| \\(h\\) | m | inclusion depth parameter |

| \\(\\theta\_{\\mathrm{deg}}\\) | degrees | horizontal orientation |

| \\(x\_0'\\) | m | inclusion-center x coordinate |

| \\(y\_0'\\) | m | inclusion-center y coordinate |



The current workflow therefore estimates inclusion size in the horizontal plane, depth, orientation, and center location.



The vertical semi-dimension \\(c\\) is currently fixed rather than sampled.



\---



\## 4. Statistical Nuisance Parameter



The seventh MCMC parameter is



\\\[

\\log\_{10}\\sigma\_{\\mathrm{strain}}.

\\]



The corresponding physical residual scale is



\\\[

\\sigma\_{\\mathrm{strain}}

=

10^{\\log\_{10}\\sigma\_{\\mathrm{strain}}}.

\\]



The complete sampled state is



\\\[

\\boxed{

\[

a,\\,

b,\\,

h,\\,

\\theta\_{\\mathrm{deg}},\\,

x\_0',\\,

y\_0',\\,

\\log\_{10}\\sigma\_{\\mathrm{strain}}

]

}

\\]



The first six entries describe the physical inclusion.



The seventh parameter describes the statistical error model.



\### Interpretation of \\(\\sigma\_{\\mathrm{strain}}\\)



This quantity should be interpreted as an \*\*effective residual strain scale\*\*, not necessarily as instrument noise alone.



It can absorb contributions from:



\- measurement uncertainty,

\- unresolved physical variability,

\- analytical-model simplification,

\- numerical/reference-model differences,

\- other model-data discrepancy.



This distinction is important when interpreting posterior predictive uncertainty.



\---



\## 5. Current Prior Bounds



The current production prior bounds are:



| Parameter | Lower | Upper |

|---|---:|---:|

| \\(a\\) | 40 m | 900 m |

| \\(b\\) | 20 m | 1000 m |

| \\(h\\) | 50 m | 900 m |

| \\(\\theta\_{\\mathrm{deg}}\\) | -90° | 90° |

| \\(x\_0'\\) | -900 m | 900 m |

| \\(y\_0'\\) | -900 m | 900 m |

| \\(\\log\_{10}\\sigma\_{\\mathrm{strain}}\\) | 0 | \\(\\log\_{10}(2000)\\) |



Thus,



\\\[

1

\\le

\\sigma\_{\\mathrm{strain}}

\\le

2000

\\]



in nanostrain.



The prior definitions belong to the production runner rather than the reusable likelihood module.



This separation allows the statistical likelihood implementation to remain reusable while prior assumptions can be changed explicitly at the experiment level.



\---



\## 6. Fixed Analytical Inputs



The current canonical 1-Darcy inversion uses:



| Quantity | Value |

|---|---:|

| \\(p\_{\\max}\\) | 999310 Pa |

| \\(E\\) | \\(8.0\\times10^9\\) Pa |

| \\(c\\) | 3.125 m |

| \\(\\nu\\) | 0.35 |

| \\(t\_{\\mathrm{peak}}\\) | 350000 s |

| \\(d\\) | 0.4 |

| \\(\\alpha\\) | 0.8 |



These quantities are not part of the current MCMC state.



They should therefore be interpreted as conditional assumptions of the current inversion.



Posterior uncertainty in the six physical parameters does \*\*not\*\* include uncertainty in these fixed quantities.



If future work promotes one or more of these quantities to uncertain parameters, the likelihood, prior definition, postprocessing, sensitivity analysis, and OED parameterization must be updated consistently.



\---



\## 7. Parameter Validation



The reusable likelihood checks basic physical/statistical validity.



The following quantities must be finite and positive:



\\\[

a,\\qquad

b,\\qquad

h,\\qquad

\\sigma\_{\\mathrm{strain}}.

\\]



The following quantities must be finite:



\\\[

\\theta\_{\\mathrm{deg}},\\qquad

x\_0',\\qquad

y\_0',\\qquad

\\log\_{10}\\sigma\_{\\mathrm{strain}}.

\\]



Detailed prior-domain enforcement remains the responsibility of the MCMC prior definitions.



Invalid sampled states receive no posterior support rather than terminating the complete sampler.



\---



\## 8. Forward Prediction



For each proposed MCMC state:



```text

\[a, b, h, theta\_deg, x0\_prime, y0\_prime, log10\_sigma\_strain]

```



the likelihood:



1\. converts the sampled vector to named parameters;

2\. computes



&#x20;  \\\[

&#x20;  \\sigma\_{\\mathrm{strain}}

&#x20;  =

&#x20;  10^{\\log\_{10}\\sigma\_{\\mathrm{strain}}};

&#x20;  \\]



3\. validates the parameter state;

4\. evaluates the analytical multi-station forward model;

5\. extracts the required strain channels;

6\. flattens them using the canonical component-major ordering;

7\. compares them with the observed vector.



The forward prediction itself is deterministic.



\---



\## 9. Strain Likelihood



Let



\\\[

\\mathbf{y}

\\]



be the observed strain vector and



\\\[

\\mathbf{f}(\\theta)

\\]



the analytical prediction.



The residual is



\\\[

\\mathbf{r}

=

\\mathbf{y}

\-

\\mathbf{f}(\\theta).

\\]



The current strain likelihood assumes independent Gaussian residuals with common standard deviation



\\\[

\\sigma\_{\\mathrm{strain}}.

\\]



For \\(N\\) observations,



\\\[

\\log L\_{\\mathrm{strain}}

=

\-\\frac{1}{2}

\\sum\_{i=1}^{N}

\\left(

\\frac{r\_i}{\\sigma\_{\\mathrm{strain}}}

\\right)^2

\-

N\\log\\sigma\_{\\mathrm{strain}}

\-

\\frac{N}{2}\\log(2\\pi).

\\]



The normalization terms are retained.



This matters because \\(\\sigma\_{\\mathrm{strain}}\\) is itself inferred. Omitting the normalization term would produce an incorrect likelihood for the residual-scale parameter.



\---



\## 10. Optional Volume Constraint



The reusable Bayesian likelihood also supports an optional volume contribution.



The analytical cuboid uses semi-dimensions



\\\[

a,\\qquad b,\\qquad c,

\\]



so its undeformed volume is



\\\[

V\_0

=

(2a)(2b)(2c)

=

8abc.

\\]



The current bulk-compressibility approximation uses



\\\[

K

=

\\frac{E}

{3(1-2\\nu)}

\\]



and



\\\[

V

=

V\_0

\\left(

1+\\frac{\\Delta P}{K}

\\right).

\\]



A Gaussian volume likelihood can be added when all required volume-observation arguments are supplied.



The default production 1-Darcy inversion is strain-only unless this optional constraint is explicitly enabled.



\---



\## 11. PyDREAM Sampling



The production runner uses PyDREAM for Markov chain Monte Carlo sampling.



Important controls include:



```text

\--max-iter

\--batch-size

\--nchains

\--rhat-threshold

\--seed

\--mp-start

```



Inspect the exact interface with:



```bash

python scripts/inversion/run\_inversion\_multi\_station.py --help

```



\### Multiprocessing



On Windows, the production workflow uses a spawn-compatible top-level likelihood wrapper.



Likelihood functions passed to multiprocessing must remain importable/pickleable.



Local nested functions should not be used as multiprocessing likelihood targets.



\---



\## 12. Batched Convergence Workflow



The inversion is executed in batches rather than assuming a fixed run length is automatically sufficient.



Conceptually:



```text

initialize chains

&#x20;     |

&#x20;     v

run PyDREAM batch

&#x20;     |

&#x20;     v

append chain histories

&#x20;     |

&#x20;     v

compute convergence diagnostics

&#x20;     |

&#x20;     +------ converged? ------ yes ---> finalize run

&#x20;     |

&#x20;     no

&#x20;     |

&#x20;     v

continue next batch

&#x20;     |

&#x20;     v

stop at max\_iter if necessary

```



The production runner therefore distinguishes:



```text

converged

```



from



```text

not\_converged

```



rather than treating sampler completion as proof of convergence.



\---



\## 13. R-hat Criterion



The current production convergence criterion is based on the Gelman-Rubin/R-hat diagnostic across all sampled parameters.



The configured default threshold is



\\\[

\\hat{R}<1.1

\\]



for every sampled parameter.



The production criterion is therefore conceptually



\\\[

\\max\_j \\hat{R}\_j < 1.1.

\\]



A run is not considered converged merely because some parameters satisfy the threshold.



The complete seven-dimensional sampled state must be considered.



\---



\## 14. Smoke Tests Versus Production Runs



A short inversion such as



```bash

python scripts/inversion/run\_inversion\_multi\_station.py \\

&#x20;   --max-iter 20 \\

&#x20;   --batch-size 20 \\

&#x20;   --nchains 4

```



is useful for verifying:



\- imports,

\- multiprocessing,

\- likelihood execution,

\- output writing,

\- run manifests,

\- convergence bookkeeping,

\- downstream postprocessing contracts.



It is \*\*not\*\* intended to produce a scientific posterior.



For example, a 20-iteration smoke run may correctly finish with



```text

run\_status = not\_converged

```



and large R-hat values.



That is a successful software test but an unsuccessful production inference.



\---



\## 15. Run Manifest



Each production run records a manifest containing the information required to reconstruct the experiment.



This includes information such as:



```text

repository root

runner

model name

station file

observed file

station names

number of stations

number of time steps

number of channels

sampled parameter names

physical parameter names

nuisance parameter names

PyDREAM settings

fixed model inputs

prior bounds

software versions

output directory

```



This manifest is part of the reproducibility contract and should be preserved with every production result.



\---



\## 16. Posterior Run Summary



The inversion also records a posterior/run summary.



Typical information includes:



```text

parameter names

chain count

iterations per chain

total samples

MAP state

posterior mean

posterior standard deviation

fixed model inputs

prior bounds

run status

convergence criterion

final R-hat values

```



The summary must retain the convergence state.



A posterior summary from a non-converged run is diagnostic rather than final scientific inference.



\---



\## 17. Bayesian Postprocessing



After a converged inversion, run:



```bash

python scripts/postprocessing/postprocess\_bayesian\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name>

```



Inspect all options with:



```bash

python scripts/postprocessing/postprocess\_bayesian\_1darcy.py --help

```



The postprocessor handles:



\- burn-in removal,

\- chain flattening,

\- convergence reporting,

\- posterior summaries,

\- posterior plots,

\- MAP predictions,

\- posterior predictive sampling,

\- residual analysis,

\- station-level metrics.



\---



\## 18. Posterior Parameter Distributions



The publication-quality posterior-distribution figure reports:



\\\[

a,\\quad

b,\\quad

h,\\quad

\\theta,\\quad

x\_0',\\quad

y\_0',\\quad

\\sigma\_{\\mathrm{strain}}.

\\]



For each parameter, the figure displays:



```text

posterior histogram

posterior mean

posterior median

MAP

95% credible interval

```



The statistical parameter is plotted in physical units as



\\\[

\\sigma\_{\\mathrm{strain}}

\\]



rather than only as



\\\[

\\log\_{10}\\sigma\_{\\mathrm{strain}}.

\\]



This improves physical interpretation while retaining the logarithmic parameterization internally for MCMC.



\---



\## 19. Parameter-Only Predictive Uncertainty



Posterior draws of



\\\[

\[a,b,h,\\theta,x\_0',y\_0']

\\]



are propagated through the analytical model.



This generates a predictive ensemble representing uncertainty caused by uncertainty in the physical inclusion parameters.



This is the \*\*parameter-only\*\* or epistemic predictive contribution.



It answers:



> How uncertain is the predicted strain because the physical inclusion itself is uncertain?



\---



\## 20. Total Predictive Uncertainty



The total predictive distribution additionally includes the inferred residual scale



\\\[

\\sigma\_{\\mathrm{strain}}.

\\]



Conceptually,



\\\[

y\_{\\mathrm{rep}}

=

f(\\theta)

\+

\\epsilon,

\\]



with



\\\[

\\epsilon

\\sim

\\mathcal{N}

(0,\\sigma\_{\\mathrm{strain}}^2).

\\]



This combines:



1\. uncertainty in the physical parameters; and

2\. unresolved model/data variability represented by the residual model.



The parameter-only and total predictive intervals should therefore not be interpreted as the same quantity.



\---



\## 21. MAP Prediction



The Maximum A Posteriori state is the sampled state with the largest posterior/log-probability value in the retained chains.



The corresponding analytical prediction is useful as a representative best-fit model.



However, the MAP curve alone does not describe posterior uncertainty.



Scientific interpretation should consider the full posterior and predictive intervals rather than only the MAP solution.



\---



\## 22. Residual Diagnostics



For observed strain



\\\[

y

\\]



and representative model prediction



\\\[

\\hat{y},

\\]



the residual is



\\\[

r

=

y-\\hat{y}.

\\]



Residual diagnostics are used to identify:



\- systematic model bias,

\- station-specific mismatch,

\- component-specific mismatch,

\- temporal structure not represented by the analytical model,

\- possible inadequacy of the independent Gaussian residual assumption.



The inferred \\(\\sigma\_{\\mathrm{strain}}\\) summarizes residual scale but does not replace residual diagnostics.



\---



\## 23. Bayesian OED Dependency



Posterior-informed OED uses the Bayesian posterior as an input distribution over plausible inclusion geometries.



Therefore the sequence



```text

Bayesian inversion

&#x20;     |

&#x20;     v

convergence verification

&#x20;     |

&#x20;     v

posterior-informed OED

```



is scientifically important.



The Bayesian OED runner normally rejects non-converged posterior runs.



The option



```text

\--allow-nonconverged

```



exists for software testing only.



It should not be used to turn a diagnostic posterior into a production scientific result.



\---



\## 24. Recommended Production Practice



For a final Bayesian analysis:



1\. use the canonical dataset;

2\. record the Git commit;

3\. record all prior bounds;

4\. preserve fixed model inputs;

5\. use multiple PyDREAM chains;

6\. use a reproducible random seed;

7\. run sufficient iterations;

8\. inspect R-hat for every sampled parameter;

9\. confirm convergence before interpretation;

10\. run the production postprocessor;

11\. inspect traces and posterior correlations;

12\. inspect station/component residuals;

13\. distinguish parameter-only from total predictive uncertainty;

14\. preserve the run manifest and convergence files;

15\. use only converged posterior results for posterior-informed OED and production EIG.



\---



\## 25. Current Production Contract



The current production Bayesian contract is:



```text

Physical parameters

&#x20;   a

&#x20;   b

&#x20;   h

&#x20;   theta\_deg

&#x20;   x0\_prime

&#x20;   y0\_prime



Nuisance parameter

&#x20;   log10\_sigma\_strain



Fixed inputs

&#x20;   pmax

&#x20;   E

&#x20;   c

&#x20;   nu

&#x20;   tpeak

&#x20;   d

&#x20;   alpha



Observed data

&#x20;   8 stations

&#x20;   107 times

&#x20;   32 strain channels

&#x20;   3424 strain observations



Sampler

&#x20;   PyDREAM

&#x20;   multiple chains

&#x20;   batched execution

&#x20;   R-hat convergence checking



Postprocessing

&#x20;   posterior summaries

&#x20;   MAP

&#x20;   credible intervals

&#x20;   parameter-only predictive uncertainty

&#x20;   total predictive uncertainty

&#x20;   residual diagnostics

```



Changes to this contract should be accompanied by corresponding updates to:



```text

source modules

production runner

postprocessor

sensitivity analysis

OED parameterization

tests

documentation

```



\---



\## 26. Related Documentation



See:



```text

docs/workflow.md

docs/sensitivity\_analysis.md

docs/optimal\_experimental\_design.md

docs/reproducibility.md

docs/palmetto.md

```



for the broader workflow, sensitivity methods, experimental-design framework, reproducibility requirements, and HPC deployment.

