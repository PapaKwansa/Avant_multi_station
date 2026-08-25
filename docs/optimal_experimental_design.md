\# Optimal Experimental Design



\## 1. Purpose



The AVANT Optimal Experimental Design (OED) framework determines which strain observations are most useful for identifying uncertain properties of a pressurized subsurface inclusion.



The framework is designed for two situations:



1\. an existing set of strainmeter stations is available and we need to determine which stations are most informative; and

2\. the inclusion geometry and boundaries are uncertain before deployment, so sensor locations themselves must be designed robustly.



The reusable OED implementation is under



```text

src/avant\_model/oed/

```



with production runners under



```text

scripts/oed/

```



The framework combines:



```text

local Fisher information

&#x20;       +

station-subset analysis

&#x20;       +

prior-robust geometry uncertainty

&#x20;       +

free spatial sensor placement

&#x20;       +

posterior-informed Bayesian design

&#x20;       +

Expected Information Gain

```



\---



\## 2. Scientific Question



Sensitivity analysis asks:



> Which physical parameters strongly affect the predicted strain?



OED asks a different question:



> Which observations or sensor configurations provide the greatest independent information about those uncertain parameters?



A station can record a large strain response and still be a poor experimental-design choice if it provides information that is highly redundant with other stations.



OED therefore considers the structure of the model Jacobian and the resulting parameter-information matrix rather than simply maximizing strain amplitude.



\---



\## 3. Inferred Parameter Space



The current six-dimensional physical parameter vector is



\\\[

\\theta

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



The current prior/reference domain is:



| Parameter | Lower | Upper |

|---|---:|---:|

| \\(a\\) | 40 m | 900 m |

| \\(b\\) | 20 m | 1000 m |

| \\(h\\) | 50 m | 900 m |

| \\(\\theta\_{\\mathrm{deg}}\\) | -90° | 90° |

| \\(x\_0'\\) | -900 m | 900 m |

| \\(y\_0'\\) | -900 m | 900 m |



The OED workflow is therefore concerned with designs that improve information about these six physical parameters.



The residual-scale nuisance parameter



\\\[

\\log\_{10}\\sigma\_{\\mathrm{strain}}

\\]



is part of the Bayesian inversion but is not the principal six-dimensional physical OED target in the current classical design framework.



\---



\## 4. Local Fisher Information



For a design \\(d\\), let



\\\[

f(\\theta;d)

\\]



denote the model prediction at the observations provided by that design.



The local sensitivity matrix is



\\\[

J\_d

=

\\frac{\\partial f(\\theta;d)}

{\\partial\\theta}.

\\]



With observation weighting matrix \\(W\_d\\), the Fisher information matrix is



\\\[

F\_d

=

J\_d^T W\_d J\_d.

\\]



The Fisher matrix summarizes how strongly the observations constrain local perturbations in the physical parameters.



\---



\## 5. Fisher Diagnostics



The implementation evaluates diagnostics including:



\- rank;

\- condition number;

\- determinant/log determinant;

\- eigenvalue structure;

\- parameter-information measures.



A full-rank Fisher matrix is necessary for local identifiability of all parameters in the tested design.



The canonical eight-station AVANT network has been verified to produce a full-rank six-parameter Fisher system at the reference geometry.



The measured local diagnostics provide a baseline against which station subsets and alternative sensor networks can be compared.



\---



\## 6. D-Optimality



One principal information criterion is D-optimality.



For a full-rank Fisher matrix,



\\\[

\\Phi\_D

=

\\log\\det(F).

\\]



Higher values indicate greater local information volume in parameter space.



Equivalently, maximizing \\(\\det(F)\\) tends to minimize the local volume of the asymptotic parameter-confidence region.



The OED framework therefore treats D-optimality as a higher-is-better metric.



\---



\## 7. Other Design Metrics



The production OED framework supports multiple design metrics.



Depending on the analysis, these can include quantities related to:



\- D-optimality;

\- conditioning;

\- parameter-specific information;

\- posterior/robust utility;

\- conservative utility;

\- worst-case performance.



Metric direction is explicitly tracked because not all useful metrics have the same optimization direction.



For example:



```text

d\_optimality     higher is better

condition\_number lower is better

```



This prevents accidental ranking reversals when comparing designs.



\---



\## 8. Existing-Station OED



The canonical network contains eight strainmeter stations:



```text

S01

S02

S03

S04

S05

S06

S07

S08

```



For \\(N\\) stations, all nonempty station subsets can be enumerated.



For the canonical eight-station network this produces



\\\[

2^8-1=255

\\]



candidate nonempty station networks.



Each subset is evaluated using the corresponding subset of the model Jacobian and observation weighting.



The objective is to determine which existing stations can be removed while preserving as much parameter information as possible.



\---



\## 9. Why Station Subsets Matter



Adding a sensor does not automatically provide proportional information.



Two stations may have:



\- similar distances to the inclusion;

\- similar azimuths;

\- similar strain responses;

\- strongly correlated sensitivity vectors.



Such stations can be redundant.



Station-subset OED makes this redundancy explicit.



The result can identify:



\- highly informative stations;

\- redundant stations;

\- minimally sufficient station networks;

\- information gained by adding additional stations.



\---



\## 10. Geometry Uncertainty



Before an inclusion has been characterized, its geometry and boundaries may be unknown.



Potentially uncertain properties include:



```text

size

shape

depth

orientation

horizontal center

extent relative to the sensor array

```



A design optimized for one assumed geometry can therefore be fragile.



The robust OED framework addresses this by evaluating each candidate design across an ensemble of plausible geometry scenarios.



\---



\## 11. Prior Geometry Scenarios



Geometry scenarios can be generated over the physical prior domain using Latin hypercube sampling.



The six-dimensional geometry state is



\\\[

\[

a,b,h,\\theta\_{\\mathrm{deg}},x\_0',y\_0'

].

\\]



LHS provides broad coverage of the prior domain without relying on a simple Cartesian grid, which would become prohibitively expensive in six dimensions.



A reproducible random seed should be recorded with production OED runs.



\---



\## 12. Boundary Scenarios



Robust design also includes deterministic boundary stress tests.



Boundary scenarios deliberately evaluate combinations near the limits of the geometry domain.



These cases are useful for answering:



> Does the chosen sensor network remain informative if the true body lies near an extreme but still physically allowed geometry?



Boundary analysis should not replace probabilistic sampling, but it is a useful complement.



\---



\## 13. Robust Design Utility



For candidate design \\(d\\), suppose the design metric is evaluated across scenario utilities



\\\[

U\_d^{(1)},\\,

U\_d^{(2)},\\,

\\ldots,\\,

U\_d^{(N)}.

\\]



The production framework summarizes performance using statistics such as:



\- expected utility;

\- median utility;

\- conservative utility;

\- worst-case utility;

\- standard deviation;

\- coefficient of variation.



A robust design should not merely perform exceptionally in one geometry while failing in many others.



The objective is a good information/robustness tradeoff.



\---



\## 14. Weighted Robust Design



Not all geometry scenarios necessarily have equal posterior or prior probability.



Weighted robust design therefore allows scenario weights



\\\[

w\_1,\\ldots,w\_N,

\\qquad

\\sum\_i w\_i=1.

\\]



The weighted expected utility is



\\\[

\\mathbb{E}\_w\[U]

=

\\sum\_i w\_i U\_i.

\\]



Weighted quantiles and weighted conservative summaries can also be calculated.



This becomes particularly important after Bayesian inversion, when the posterior provides nonuniform probabilities over plausible geometries.



\---



\## 15. Free Spatial Sensor Placement



When new strainmeters can be deployed, the design problem becomes spatial rather than merely combinatorial.



Candidate locations can be generated from a user-defined rectangular region.



For example:



```text

x = \[xmin, xmax]

y = \[ymin, ymax]

```



can be discretized into a candidate grid.



The sensor-placement module can then construct candidate networks of a specified size.



\---



\## 16. Sensor Spacing



A realistic deployment must respect minimum spacing between sensors.



The placement framework therefore supports network spacing diagnostics.



A candidate network with coincident or nearly coincident stations may be rejected or assigned poor design utility.



This prevents a mathematical optimum from becoming an impractical deployment.



\---



\## 17. Greedy Spatial Selection



A geometry-based greedy strategy can construct a network by selecting stations that maximize spatial separation while satisfying deployment constraints.



Greedy selection is computationally cheaper than exhaustive enumeration over a large candidate grid.



It is useful for generating strong candidate networks that can subsequently be evaluated with the information-based OED criteria.



Greedy placement is a candidate-generation strategy, not a guarantee of global information optimality.



\---



\## 18. Random Candidate Networks



Random candidate networks provide another way to sample the design space.



They are useful for:



\- exploratory comparisons;

\- generating diverse candidate designs;

\- providing initial candidate pools;

\- validating optimization code.



A fixed random seed should be used when results need to be reproduced.



\---



\## 19. Posterior-Informed OED



After Bayesian inversion, the posterior distribution provides updated information about the likely inclusion geometry.



Posterior-informed OED uses posterior samples as geometry scenarios.



The workflow becomes



```text

existing strain observations

&#x20;            |

&#x20;            v

&#x20;     Bayesian posterior

&#x20;            |

&#x20;            v

&#x20;  plausible geometry states

&#x20;            |

&#x20;            v

&#x20;      candidate designs

&#x20;            |

&#x20;            v

&#x20;  posterior-weighted utility

```



This is different from prior-robust OED because designs are no longer optimized across the entire initial prior equally.



The goal is to improve the experiment based on what is already known.



\---



\## 20. Posterior Existing-Network Design



For an existing network, posterior-informed OED evaluates candidate station subsets under multiple posterior geometries.



For each retained posterior scenario:



1\. evaluate the analytical model;

2\. compute the Jacobian;

3\. evaluate each candidate station subset;

4\. calculate the selected design metric;

5\. aggregate performance across posterior scenarios.



The resulting ranking identifies which existing station network is most informative under the current posterior uncertainty.



\---



\## 21. Posterior Spatial Placement



The same posterior scenario concept can be applied to free spatial placement.



Candidate locations are generated across the permitted deployment region.



Candidate networks are then evaluated across posterior geometry states.



This asks:



> Given what we currently believe about the inclusion, where should additional strainmeters be deployed?



This is a more targeted question than prior-robust placement.



\---



\## 22. Expected Information Gain



Expected Information Gain (EIG) provides a fully Bayesian utility.



For a design \\(d\\),



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



It measures the expected reduction in uncertainty produced by the future experiment.



Higher EIG indicates a more informative candidate experiment.



Unlike local Fisher OED, EIG directly uses a probabilistic parameter distribution and a probabilistic observation model.



\---



\## 23. Nested Monte Carlo EIG



The current EIG implementation uses a nested Monte Carlo approximation.



Conceptually:



```text

Outer posterior samples

&#x20;       |

&#x20;       v

predict future observations

&#x20;       |

&#x20;       v

Inner likelihood/posterior calculations

&#x20;       |

&#x20;       v

estimate information gain

```



The production estimate therefore depends on:



```text

posterior sample count

outer Monte Carlo count

inner Monte Carlo count

candidate design count

```



The corresponding command-line controls include:



```text

\--posterior-samples

\--eig-outer

\--eig-inner

\--eig-designs

```



Small values are appropriate for software validation only.



Final EIG comparisons should use substantially larger Monte Carlo ensembles and should assess numerical stability.



\---



\## 24. EIG Units



The implementation reports EIG in:



```text

nats

bits

```



Conversion is



\\\[

\\mathrm{EIG}\_{\\mathrm{bits}}

=

\\frac{

\\mathrm{EIG}\_{\\mathrm{nats}}

}{

\\ln 2

}.

\\]



EIG values should be compared under the same likelihood, parameterization, prior/posterior definition, and sampling configuration.



\---



\## 25. EIG Monte Carlo Error



Monte Carlo EIG estimates contain sampling error.



The implementation therefore records quantities such as an estimated standard error.



A large estimated standard error relative to the difference between two candidate designs means the ranking may not be statistically stable.



Therefore:



> A numerical EIG ranking should not be treated as definitive unless the Monte Carlo uncertainty is small enough to distinguish the leading candidates.



Increasing outer and inner sample counts should stabilize the estimate.



\---



\## 26. Current OED Workflow



A complete OED analysis can be organized as:



```text

&#x20;                   Unknown geometry

&#x20;                         |

&#x20;                         v

&#x20;               Prior geometry ensemble

&#x20;                         |

&#x20;                         v

&#x20;                Classical robust OED

&#x20;                         |

&#x20;              +----------+----------+

&#x20;              |                     |

&#x20;              v                     v

&#x20;     Existing station sets   Free spatial placement

&#x20;              |                     |

&#x20;              +----------+----------+

&#x20;                         |

&#x20;                         v

&#x20;                Bayesian inversion

&#x20;                         |

&#x20;                         v

&#x20;                   Posterior

&#x20;                         |

&#x20;                         v

&#x20;               Posterior-informed

&#x20;                  Bayesian OED

&#x20;                         |

&#x20;              +----------+----------+

&#x20;              |                     |

&#x20;              v                     v

&#x20;         Existing subset       Spatial placement

&#x20;              |                     |

&#x20;              +----------+----------+

&#x20;                         |

&#x20;                         v

&#x20;                        EIG

```



This hierarchy allows experimental design to evolve as the amount of information about the subsurface target increases.



\---



\## 27. Recommended Design Strategy When Geometry Is Unknown



When the inclusion shape and boundaries are initially unknown, the recommended workflow is:



\### Stage 1 — broad robust design



Use prior geometry scenarios and boundary cases.



The objective is to avoid choosing a network that is excellent only for one speculative geometry.



\### Stage 2 — acquire initial observations



Use the resulting robust network to collect strain observations.



\### Stage 3 — Bayesian inversion



Infer the inclusion geometry and quantify posterior uncertainty.



\### Stage 4 — adaptive design



Use the posterior to redesign or augment the network.



\### Stage 5 — EIG refinement



Use EIG when sufficient computational resources are available and the posterior is sufficiently well characterized.



This creates an adaptive experiment:



\\\[

\\text{prior}

\\rightarrow

\\text{experiment}

\\rightarrow

\\text{posterior}

\\rightarrow

\\text{next experiment}.

\\]



\---



\## 28. Design Size



A larger network usually contains more observations and can therefore provide more total information.



However, increasing station count has costs:



\- instrumentation;

\- installation;

\- maintenance;

\- communications;

\- deployment complexity;

\- data processing.



OED should therefore compare designs at multiple network sizes rather than assuming that “more stations” is automatically optimal.



Useful comparisons include:



```text

2 stations

3 stations

4 stations

5 stations

...

8 stations

```



or the relevant deployment budget.



\---



\## 29. Robustness Versus Optimality



The single highest-scoring design under one nominal geometry is not necessarily the best practical design.



A robust design may have slightly lower expected information but substantially better worst-case performance.



For uncertain targets, a modest sacrifice in nominal optimality may be justified by improved resilience to geometry misspecification.



The production framework therefore reports multiple utility summaries rather than relying solely on one score.



\---



\## 30. Sensor Placement Interpretation



A useful placement pattern generally provides diversity in:



\- radial distance from the target;

\- azimuth;

\- orientation relative to the expected inclusion axes;

\- sensitivity to different strain components.



Stations that occupy similar geometric positions relative to the target may be redundant even when their predicted strain amplitudes are large.



The optimal configuration should therefore be interpreted through its information structure, not just by visually inspecting strain amplitude.



\---



\## 31. Classical OED Versus Bayesian OED



\### Classical/local OED



Uses local derivatives around a nominal parameter state.



Strengths:



\- fast;

\- interpretable;

\- useful for identifiability diagnostics;

\- suitable for early design screening.



Limitations:



\- local;

\- dependent on the nominal geometry;

\- may not capture strong nonlinearity.



\### Robust OED



Evaluates multiple prior geometry scenarios.



Strengths:



\- protects against geometry misspecification;

\- useful before substantial data are available;

\- supports conservative deployment choices.



\### Bayesian OED



Uses the current posterior.



Strengths:



\- reflects information already learned;

\- adaptive;

\- naturally incorporates posterior uncertainty.



\### EIG



Uses a fully Bayesian future-observation criterion.



Strengths:



\- directly targets expected uncertainty reduction;

\- supports nonlinear/posterior-informed design.



Limitations:



\- computationally expensive;

\- Monte Carlo uncertainty must be controlled.



\---



\## 32. Production Commands



\### Local Fisher OED



```bash

python scripts/oed/run\_oed\_1darcy.py \\

&#x20;   --mode local

```



\### Existing-station OED



```bash

python scripts/oed/run\_oed\_1darcy.py \\

&#x20;   --mode existing

```



\### Prior-robust OED



```bash

python scripts/oed/run\_oed\_1darcy.py \\

&#x20;   --mode robust

```



\### Boundary stress testing



```bash

python scripts/oed/run\_oed\_1darcy.py \\

&#x20;   --mode boundary

```



\### Free spatial placement



```bash

python scripts/oed/run\_oed\_1darcy.py \\

&#x20;   --mode placement

```



Inspect the exact current arguments with:



```bash

python scripts/oed/run\_oed\_1darcy.py --help

```



\---



\## 33. Bayesian OED Commands



Validate a Bayesian run first:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode validate

```



Posterior-informed existing-network design:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode posterior\_existing

```



Posterior-informed placement:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode posterior\_placement

```



Expected Information Gain:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode eig

```



Use:



```text

\--allow-nonconverged

```



only for software testing or diagnostic experiments.



Production Bayesian OED should use a converged posterior.



\---



\## 34. Smoke Tests



Small configurations are useful for verifying the software.



For example:



```text

posterior samples = 2–4

candidate networks = 2–3

EIG outer samples = 3

EIG inner samples = 4

```



These settings can produce fast execution and confirm that the computational pathway is correct.



They are not appropriate for final scientific EIG ranking.



Similarly, a small number of posterior samples can be used to validate station-subset or placement code without representing the full posterior distribution.



\---



\## 35. Production OED Requirements



Final OED analyses should document:



```text

Git commit

dataset

station metadata

physical parameter bounds

reference geometry or posterior source

number of geometry scenarios

scenario weights

candidate design definition

network size

minimum spacing

design metric

noise model

Monte Carlo sample counts

random seed

software environment

```



For Bayesian OED, additionally record:



```text

posterior source

posterior convergence state

posterior sample selection

representative sigma rule

EIG outer sample count

EIG inner sample count

EIG numerical uncertainty

```



\---



\## 36. Interpreting an OED Ranking



A design ranking should always be interpreted together with:



\- design size;

\- metric direction;

\- scenario count;

\- scenario weighting;

\- worst-case behavior;

\- variability across scenarios;

\- observation noise;

\- computational uncertainty.



A design with the highest expected utility but very poor worst-case utility may not be preferable to a slightly lower but substantially more robust design.



For Bayesian EIG, differences should be interpreted relative to Monte Carlo standard error.



\---



\## 37. Current Scientific Interpretation



The OED framework is not merely a station-ranking utility.



It provides a framework for deciding how to collect better data when the subsurface target is only partially known.



The central logic is:



\\\[

\\boxed{

\\text{uncertain geometry}

\\rightarrow

\\text{robust observations}

\\rightarrow

\\text{Bayesian inference}

\\rightarrow

\\text{adaptive design}

}

\\]



This is particularly important for a target whose:



\- shape is uncertain;

\- dimensions are uncertain;

\- boundaries are uncertain;

\- orientation is uncertain;

\- center is uncertain; and

\- depth is uncertain.



The purpose of robust and Bayesian OED is therefore to reduce the risk of designing an experiment around an incorrect geometric assumption.



\---



\## 38. Related Documentation



See:



```text

docs/workflow.md

docs/bayesian\_inversion.md

docs/sensitivity\_analysis.md

docs/reproducibility.md

docs/palmetto.md

```



for the complete analytical workflow, Bayesian inference, sensitivity analysis, reproducibility requirements, and HPC execution.

