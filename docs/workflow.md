\# AVANT Production Workflow



\## Purpose



This document describes the end-to-end production workflow for the AVANT multi-station analytical poroelastic strain model.



The workflow connects:



1\. canonical COMSOL strain data,

2\. the analytical half-space forward model,

3\. Bayesian parameter inference,

4\. posterior uncertainty quantification,

5\. sensitivity analysis,

6\. classical and robust optimal experimental design,

7\. posterior-informed Bayesian experimental design, and

8\. Expected Information Gain.



The objective is not only to estimate the properties of a pressurized subsurface inclusion, but also to quantify what can and cannot be inferred from the available strain observations and determine how future strainmeter deployments can improve parameter identification.



\---



\## 1. Physical Problem



A pore-pressure perturbation within a subsurface inclusion generates poroelastic deformation.



The analytical formulation combines Eshelby eigenstrain theory with Mindlin half-space solutions and the corresponding Chen equations to represent deformation in an elastic half-space with a traction-free surface.



The inclusion is described geometrically by the semi-dimensions



\\\[

a,\\qquad b,\\qquad c,

\\]



depth parameter



\\\[

h,

\\]



horizontal orientation



\\\[

\\theta,

\\]



and horizontal center



\\\[

(x\_0',y\_0').

\\]



The analytical model predicts displacement and strain at arbitrary observation coordinates.



For the current AVANT multi-station inversion, the observational strain vector uses



\\\[

\\epsilon\_{xx},\\qquad

\\epsilon\_{yy},\\qquad

\\epsilon\_{zz},\\qquad

\\epsilon\_{xy}.

\\]



\---



\## 2. Canonical 1-Darcy Dataset



The production workflow uses the canonical transient COMSOL dataset under



```text

datasets/comsol/1darcy/

```



with station metadata in



```text

datasets/comsol/1darcy/metadata/stations.csv

```



and processed strain observations in



```text

datasets/comsol/1darcy/processed/strain.csv

```



The canonical data contract is:



```text

stations            8

station names       S01 ... S08

time steps          107

strain components   4

strain channels     32

observations        3424

```



The four components are stored for every station.



The observed and modeled data vectors must use identical ordering. The production likelihood uses component-major ordering:



```text

eXX: all stations

eYY: all stations

eXY: all stations

eZZ: all stations

```



with each station time series concatenated in the specified column order.



This ordering is part of the scientific data contract and is regression-tested.



\---



\## 3. Analytical Forward Model



The reusable forward-model implementation is located under



```text

src/avant\_model/model/

```



The principal entry point is



```text

forward\_model\_multi\_station.py

```



Supporting modules implement coordinate transformations, strain calculations, and strain-tensor rotation.



For each parameter state, the forward model evaluates the transient strain response at all stations and returns the modeled strain components using the same station/component convention as the observed dataset.



The forward model is deterministic for a fixed parameter state.



Regression tests verify:



\- output dimensions,

\- channel naming,

\- finite predictions,

\- time alignment,

\- deterministic behavior,

\- tensor-rotation invariants,

\- the adopted shear-strain sign convention.



\---



\## 4. Production Parameterization



The current Bayesian inversion estimates six physical parameters:



\\\[

\\theta\_{\\mathrm{physical}}

=

\[a,b,h,\\theta\_{\\mathrm{deg}},x\_0',y\_0'].

\\]



Their roles are:



| Parameter | Meaning |

|---|---|

| \\(a\\) | inclusion semi-dimension along the local x direction |

| \\(b\\) | inclusion semi-dimension along the local y direction |

| \\(h\\) | inclusion depth parameter |

| \\(\\theta\\) | horizontal inclusion orientation |

| \\(x\_0'\\) | horizontal inclusion-center x coordinate |

| \\(y\_0'\\) | horizontal inclusion-center y coordinate |



The current analytical inversion holds several model properties fixed, including the vertical semi-dimension \\(c\\), elastic properties, and transient pressure-model parameters.



The production MCMC state also contains



\\\[

\\log\_{10}\\sigma\_{\\mathrm{strain}},

\\]



so that



\\\[

\\theta\_{\\mathrm{MCMC}}

=

\[

a,b,h,\\theta\_{\\mathrm{deg}},x\_0',y\_0',

\\log\_{10}\\sigma\_{\\mathrm{strain}}

].

\\]



The physical residual scale is



\\\[

\\sigma\_{\\mathrm{strain}}

=

10^{\\log\_{10}\\sigma\_{\\mathrm{strain}}}.

\\]



This parameter represents the effective residual uncertainty between the observed strain data and analytical predictions.



It can include contributions from:



\- measurement uncertainty,

\- unresolved physical variability,

\- numerical/reference-model differences,

\- structural discrepancy of the analytical approximation.



It should therefore not automatically be interpreted as instrument noise alone.



\---



\## 5. Bayesian Inversion



The reusable statistical model is implemented in



```text

src/avant\_model/inversion/bayesian\_inversion\_multi\_station.py

```



and the production command-line runner is



```text

scripts/inversion/run\_inversion\_multi\_station.py

```



For one parameter state, the workflow is



```text

sample parameter state

&#x20;       |

&#x20;       v

validate physical parameters

&#x20;       |

&#x20;       v

run analytical forward model

&#x20;       |

&#x20;       v

flatten predicted strain

&#x20;       |

&#x20;       v

observed - predicted residual

&#x20;       |

&#x20;       v

Gaussian strain likelihood

&#x20;       |

&#x20;       v

PyDREAM posterior sampling

```



For strain residual vector \\(r\\) and effective residual standard deviation

\\(\\sigma\_{\\mathrm{strain}}\\), the Gaussian log likelihood is



\\\[

\\log L

=

\-\\frac{1}{2}

\\sum\_i

\\left(

\\frac{r\_i}{\\sigma\_{\\mathrm{strain}}}

\\right)^2

\-

N\\log\\sigma\_{\\mathrm{strain}}

\-

\\frac{N}{2}\\log(2\\pi).

\\]



The inversion runner owns the prior definitions and MCMC controls. The reusable likelihood module intentionally does not define priors or sampler settings.



\---



\## 6. Convergence



A successful program execution is not equivalent to a converged Bayesian inversion.



The production workflow evaluates convergence across the PyDREAM chains using the configured Gelman-Rubin/R-hat criterion.



A short run such as



```text

20 iterations per chain

```



is a software smoke test only.



A posterior marked



```text

not\_converged

```



must not be interpreted as final scientific inference.



Production posterior interpretation requires satisfactory convergence diagnostics.



The run directory preserves the convergence state so that postprocessing and Bayesian OED can distinguish diagnostic runs from scientifically usable posterior results.



\---



\## 7. Posterior Postprocessing and Uncertainty



The production postprocessor is



```text

scripts/postprocessing/postprocess\_bayesian\_1darcy.py

```



It converts raw chain histories into interpretable posterior diagnostics.



Outputs include:



\- chain traces,

\- marginal posterior distributions,

\- posterior means,

\- medians,

\- MAP estimates,

\- credible intervals,

\- posterior correlations,

\- pairwise parameter relationships,

\- station-level observed-versus-modeled comparisons,

\- residual diagnostics,

\- posterior predictive intervals,

\- quantitative predictive metrics.



Two uncertainty concepts are kept distinct.



\### Parameter uncertainty



Variation in



\\\[

\[a,b,h,\\theta,x\_0',y\_0']

\\]



represents uncertainty about the physical inclusion.



This produces parameter-only posterior predictive uncertainty.



\### Residual/model-discrepancy uncertainty



The inferred



\\\[

\\sigma\_{\\mathrm{strain}}

\\]



represents unresolved observed-model variability.



Combining parameter uncertainty with this residual contribution produces the total predictive distribution.



Keeping these quantities separate makes it possible to distinguish uncertainty about the inclusion from mismatch that the simplified analytical model cannot resolve explicitly.



\---



\## 8. Sensitivity Analysis



The production sensitivity runner is



```text

scripts/sensitivity/run\_sensitivity\_1darcy.py

```



Sensitivity analysis is used to understand how changes in the six physical inversion parameters affect predicted strain and model-data mismatch.



The workflow includes complementary methods:



\### One-at-a-time analysis



Each parameter is varied while the others remain at a reference state.



This provides interpretable local response curves.



\### Two-dimensional response surfaces



Pairs of parameters are varied simultaneously.



These surfaces reveal parameter interactions, tradeoffs, and potentially non-unique regions.



\### Global sampling



The full parameter domain is sampled to evaluate behavior away from the reference solution.



\### Sobol analysis



Variance-based Sobol indices quantify first-order and total parameter contributions over the specified parameter domain.



Sensitivity analysis describes model response. It should not be interpreted as identical to Bayesian identifiability or experimental-design information.



\---



\## 9. Classical Fisher OED



The reusable OED package is located under



```text

src/avant\_model/oed/

```



Local OED starts from the model Jacobian



\\\[

J

=

\\frac{\\partial f}{\\partial\\theta}.

\\]



For observation weighting matrix \\(W\\),



\\\[

F

=

J^T W J

\\]



defines the Fisher information matrix.



The Fisher system provides local diagnostics such as:



\- matrix rank,

\- condition number,

\- determinant/log determinant,

\- parameter-information measures,

\- design comparison metrics.



The canonical eight-station reference network has been verified to provide a full-rank six-parameter Fisher system at the reference geometry.



\---



\## 10. Existing-Station Design



For an existing monitoring network, OED can evaluate subsets of the available stations.



The question becomes:



> Which subset of the current stations retains the most useful information about the uncertain physical parameters?



The implementation preserves the component-major observation ordering when extracting station subsets from the full Jacobian.



Candidate subsets can be compared by design size and information criterion.



\---



\## 11. Geometry-Uncertain Robust OED



Before an inclusion has been characterized, its size, depth, orientation, center, and boundaries may be uncertain.



A sensor network optimized only for one assumed geometry may therefore perform poorly if that geometry is wrong.



The robust OED workflow evaluates candidate designs across ensembles of plausible inclusion geometries.



Geometry scenarios can be constructed from:



\- Latin hypercube prior sampling,

\- random prior sampling,

\- deterministic prior-boundary cases,

\- Bayesian posterior samples.



Designs can then be summarized using:



\- expected utility,

\- median utility,

\- conservative utility,

\- worst-case utility,

\- variability/stability.



This provides a practical strategy for placing strainmeters when the target body's geometry is not known before deployment.



\---



\## 12. Free Spatial Sensor Placement



The sensor-placement module supports candidate monitoring locations that are not restricted to the existing AVANT stations.



Candidate locations may be generated using:



\- rectangular grids,

\- random spatial sampling,

\- deployment bounds,

\- excluded regions,

\- minimum sensor-spacing constraints.



Candidate networks can be generated through exhaustive enumeration, random sampling, or geometry-based greedy selection.



The resulting networks can then be evaluated using the same Fisher, robust, or Bayesian design criteria.



\---



\## 13. Posterior-Informed Bayesian OED



After Bayesian inversion, the current posterior provides a probability distribution over plausible inclusion geometries.



Instead of treating all prior geometries equally, posterior-informed OED evaluates designs using the uncertainty that remains after assimilating the existing observations.



The workflow therefore becomes



```text

existing observations

&#x20;       |

&#x20;       v

Bayesian posterior

&#x20;       |

&#x20;       v

posterior geometry scenarios

&#x20;       |

&#x20;       v

candidate sensor networks

&#x20;       |

&#x20;       v

posterior-weighted design utility

```



This allows future monitoring to be adapted to what has already been learned.



\---



\## 14. Expected Information Gain



Expected Information Gain provides a fully Bayesian design criterion.



For proposed design \\(d\\), EIG asks how much the hypothetical future observations are expected to reduce uncertainty about the parameters.



Conceptually,



\\\[

\\mathrm{EIG}(d)

=

\\mathbb{E}

\\left\[

D\_{\\mathrm{KL}}

\\left(

p(\\theta\\mid y,d)

\\parallel

p(\\theta)

\\right)

\\right].

\\]



Larger EIG indicates greater expected information.



The implementation uses nested Monte Carlo approximation, so production EIG calculations require substantially more sampling than software smoke tests.



Small values of



```text

posterior-samples

eig-outer

eig-inner

```



are appropriate for verifying execution but not for reporting final scientific EIG rankings.



\---



\## 15. Recommended Production Sequence



The recommended end-to-end analysis is



```text

1\. Validate canonical dataset

&#x20;           |

&#x20;           v

2\. Validate analytical forward model

&#x20;           |

&#x20;           v

3\. Run Bayesian inversion

&#x20;           |

&#x20;           v

4\. Verify convergence

&#x20;           |

&#x20;           v

5\. Postprocess posterior

&#x20;           |

&#x20;           v

6\. Run sensitivity analysis

&#x20;           |

&#x20;           v

7\. Run classical / robust OED

&#x20;           |

&#x20;           v

8\. Run posterior-informed OED

&#x20;           |

&#x20;           v

9\. Run production EIG

&#x20;           |

&#x20;           v

10\. Archive run metadata and figures

```



Bayesian postprocessing and posterior-informed OED should normally use a converged posterior.



Classical prior-based OED can be performed independently of Bayesian convergence.



\---



\## 16. Local Versus HPC Execution



Fast deterministic calculations are appropriate for local execution, including:



\- unit/regression tests,

\- data-contract checks,

\- forward-model smoke tests,

\- local Fisher diagnostics,

\- small OED checks.



Computationally intensive analyses are intended for HPC execution, including:



\- long PyDREAM chains,

\- large Sobol analyses,

\- dense response surfaces,

\- large robust geometry ensembles,

\- large candidate placement searches,

\- high-resolution EIG.



The current production HPC target is Clemson University's Palmetto cluster.



See



```text

docs/palmetto.md

```



for the deployment workflow.



\---



\## 17. Regression Testing



The public repository contains deterministic tests for the contracts most likely to affect scientific reproducibility.



The current baseline contains 60 tests covering:



```text

canonical data contract

analytical forward model

tensor rotation

Bayesian likelihood

Fisher information

geometry uncertainty

robust design

sensor placement

Bayesian design utilities

```



Run:



```bash

python -m pytest -q

```



Production MCMC and large Monte Carlo analyses are deliberately excluded from the normal test suite.



\---



\## 18. Reproducibility Principle



Every production result should be traceable to:



```text

source-code commit

dataset

station metadata

parameter definitions

prior bounds

fixed analytical inputs

random seed

software environment

analysis command

convergence state

output manifest

```



This separation between source code and generated analysis results is intentional.



Generated results are not committed to the source repository. Instead, the run metadata should provide enough information to reproduce them.



See



```text

docs/reproducibility.md

```



for the detailed reproducibility contract.

