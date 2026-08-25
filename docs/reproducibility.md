\# Reproducibility



\## 1. Purpose



Reproducibility is a core requirement of the AVANT production workflow.



Every scientific result should be traceable to:



```text

source-code version

\+

input dataset

\+

model configuration

\+

prior specification

\+

random seed

\+

software environment

\+

execution command

\+

convergence state

\+

generated outputs

```



The goal is that a future analysis can determine exactly:



\- which code was used;

\- which observations were used;

\- which assumptions were fixed;

\- which uncertain parameters were sampled;

\- how the computation was configured;

\- whether the Bayesian result actually converged; and

\- which figures and tables came from that run.



\---



\## 2. Reproducibility Layers



The AVANT workflow has five reproducibility layers:



```text

1\. Source code

&#x20;      |

&#x20;      v

2\. Data contract

&#x20;      |

&#x20;      v

3\. Model/configuration contract

&#x20;      |

&#x20;      v

4\. Execution record

&#x20;      |

&#x20;      v

5\. Analysis outputs

```



All five are required for a scientifically interpretable result.



\---



\## 3. Source-Code Version



Every production result should record the Git commit used to generate it.



From the repository root:



```bash

git rev-parse HEAD

```



Recommended record:



```text

repository:

&#x20;   Avant\_multi\_station



git commit:

&#x20;   <full commit hash>



branch:

&#x20;   <branch name>

```



The commit hash identifies the exact source state used for the analysis.



A production result should never be identified only by a vague description such as:



```text

"latest version"

```



or



```text

"August 2026 code"

```



because the source tree may continue to change.



\---



\## 4. Canonical Dataset



The production 1-Darcy workflow uses:



```text

datasets/comsol/1darcy/metadata/stations.csv

datasets/comsol/1darcy/processed/strain.csv

```



The canonical data contract is:



```text

8 stations

S01 ... S08



107 time steps



4 strain components per station



32 strain channels



3424 observations

```



The station metadata and observed strain file must correspond to one another.



The production workflow should not silently substitute another file with a similar name.



This requirement is especially important because earlier development versions contained multiple station CSV files with different station definitions.



The canonical paths above are therefore part of the reproducibility contract.



\---



\## 5. Raw Data Provenance



The current 1-Darcy dataset also contains source/reference files under:



```text

datasets/comsol/1darcy/raw/

```



and associated pressure/reference information.



Before a final public release, the provenance and redistribution status of each raw file should be documented.



A production analysis should record:



```text

source file

source location

processing date

processing script

processing options

resulting processed file

```



\---



\## 6. Data Preparation



The supported preprocessing entry point is:



```text

scripts/preprocessing/clean\_dataset.py

```



Inspect its interface with:



```bash

python scripts/preprocessing/clean\_dataset.py --help

```



The preprocessing workflow should produce the canonical station metadata and strain dataset expected by the downstream analysis.



A final reproducible workflow should preserve the command used to produce the processed data.



If the processed dataset is already supplied as part of the release, the preprocessing step may be skipped, but its provenance should remain documented.



\---



\## 7. Model Configuration



Every production result should record the fixed analytical-model inputs.



Current canonical values include:



```text

pmax      = 999310 Pa

E         = 8.0e9 Pa

c         = 3.125 m

nu        = 0.35

tpeak     = 350000 s

d         = 0.4

alpha     = 0.8

```



These are assumptions of the current canonical analysis.



Changing any of these values creates a different model configuration even if the source code is unchanged.



\---



\## 8. Bayesian Parameterization



The current physical parameter state is:



\\\[

\[

a,\\,

b,\\,

h,\\,

\\theta\_{\\mathrm{deg}},\\,

x\_0',\\,

y\_0'

].

\\]



The full sampled state additionally contains:



\\\[

\\log\_{10}\\sigma\_{\\mathrm{strain}}.

\\]



The production prior bounds are:



```text

a                \[40, 900]

b                \[20, 1000]

h                \[50, 900]

theta\_deg        \[-90, 90]

x0\_prime         \[-900, 900]

y0\_prime         \[-900, 900]

log10\_sigma      \[0, log10(2000)]

```



These bounds must be recorded with every production Bayesian analysis.



\---



\## 9. Random Seeds



Any analysis using random sampling should record its random seed.



This includes:



\- PyDREAM sampling;

\- global sensitivity sampling;

\- Latin hypercube sampling;

\- random sensor-network generation;

\- posterior subsampling;

\- EIG Monte Carlo calculations.



A fixed seed improves reproducibility.



However, reproducibility of a random stream does not guarantee identical results across all software/hardware environments if underlying numerical libraries differ.



Therefore the seed should always be recorded together with the software environment.



\---



\## 10. Bayesian Run Manifest



The production inversion writes a run manifest.



This should preserve information such as:



```text

created time

repository root

runner

model name



station file

observed file

station names

station count

time-step count

strain-channel count



parameter names

physical parameter names

nuisance parameter names



PyDREAM settings

chain count

iteration limits

batch size

R-hat threshold

random seed

multiprocessing mode



fixed model inputs

prior bounds



Python version

platform

NumPy version

pandas version



output directory

```



The run manifest should be preserved alongside the posterior results.



\---



\## 11. Convergence Record



A Bayesian production run must preserve its convergence state.



The most important fields include:



```text

run\_status

criterion

threshold

converged

iterations associated with convergence

final R-hat values

```



A run marked:



```text

not\_converged

```



is a software-execution result, not a scientifically accepted posterior.



Convergence state should therefore be visible in the output metadata rather than inferred from whether the program terminated successfully.



\---



\## 12. Smoke Runs



Smoke runs are intentionally small.



For example:



```text

4 chains

20 iterations per chain

```



are useful for checking:



\- imports;

\- likelihood evaluation;

\- multiprocessing;

\- file creation;

\- chain-history writing;

\- convergence bookkeeping;

\- downstream postprocessing;

\- Bayesian-OED contracts.



Smoke runs are not production analyses.



Their outputs should be clearly labeled or kept under a diagnostic run name.



The current Windows smoke run is an example of this principle: it successfully exercised the workflow but was explicitly marked `not\_converged`.



\---



\## 13. Production Bayesian Runs



A production run should record:



```text

model name

chain count

maximum iterations

batch size

R-hat threshold

seed

multiprocessing mode

```



The run should continue until either:



```text

all convergence criteria are satisfied

```



or



```text

the configured production limit is reached

```



If the limit is reached without convergence, the resulting posterior must remain classified as non-converged.



\---



\## 14. Posterior Retention



Posterior samples used for final analysis should be retained together with their metadata.



At minimum, preserve:



```text

parameter names

chain count

iterations

burn-in treatment

flattening convention

sample count

MAP state

posterior mean

posterior standard deviation

credible intervals

convergence state

```



When posterior samples are subsampled for Bayesian OED or plotting, the selection strategy should also be recorded.



\---



\## 15. Posterior Postprocessing



The production postprocessor is:



```text

scripts/postprocessing/postprocess\_bayesian\_1darcy.py

```



The command used to generate final figures should be recorded.



The resulting directory should identify which posterior run was processed.



The postprocessor output should include both:



```text

posterior parameter uncertainty

```



and



```text

predictive uncertainty

```



so that these quantities are not accidentally conflated.



\---



\## 16. Sensitivity Reproducibility



Sensitivity analyses should record:



```text

parameter bounds

reference state

fixed analytical inputs

OAT point count

response-surface resolution

global sample count

Sobol base sample count

random seed

```



For example:



```text

\--oat-points

\--heatmap-points

\--global-samples

\--sobol-base

```



must be part of the analysis record.



Two sensitivity analyses using the same source code but different sample sizes are not necessarily equivalent.



\---



\## 17. Classical OED Reproducibility



Classical OED should record:



```text

reference geometry

parameter bounds

observation weighting/noise model

station list

candidate subset definition

candidate placement region

network size

minimum spacing

design metric

scenario count

scenario-generation method

random seed where applicable

```



This information is needed to interpret design rankings.



\---



\## 18. Robust Geometry OED



Robust OED requires additional scenario information.



Record:



```text

scenario-generation method

number of scenarios

prior bounds

boundary-case definition

scenario weights

random seed

design metric

```



For weighted analyses, preserve the actual scenario weights used.



A robust design score without its scenario definition is incomplete.



\---



\## 19. Bayesian OED Reproducibility



Posterior-informed OED must additionally record:



```text

source Bayesian run

posterior convergence state

number of retained posterior samples

posterior sampling rule

representative sigma definition

candidate design set

scenario weights

design metric

```



A Bayesian OED result should always point back to the exact posterior run from which its geometry scenarios were derived.



\---



\## 20. EIG Reproducibility



EIG results require additional Monte Carlo settings.



Record:



```text

posterior sample count

number of designs

outer Monte Carlo count

inner Monte Carlo count

observation dimension

noise model

random seed

EIG units

estimated Monte Carlo standard error

```



The ranking should be interpreted together with its numerical error.



If two candidate designs have nearly identical EIG values but large Monte Carlo uncertainty, the ranking should be treated as unresolved.



\---



\## 21. Software Environment



At minimum record:



```text

Python version

NumPy version

pandas version

SciPy version

Matplotlib version

PyDREAM version

SALib version

pytest version

operating system

```



For HPC runs, also record:



```text

cluster

partition

job ID

node count

CPU allocation

memory allocation

walltime

module/environment configuration

```



The environment is part of the computational experiment.



\---



\## 22. Palmetto Reproducibility



Palmetto production runs should additionally preserve:



```text

SLURM job ID

job script

environment/module configuration

repository commit

working directory

output directory

start time

end time

exit status

```



A recommended production output hierarchy is:



```text

$HOME/pflotran\_surrogate\_results/

```



for other unrelated workflows, but the AVANT inversion results should use a dedicated AVANT-specific results hierarchy rather than mixing with unrelated projects.



The exact production directory will be defined in:



```text

docs/palmetto.md

```



\---



\## 23. Directory Organization for Results



Generated numerical results should not be mixed with source code.



Recommended local organization:



```text

results/

├── bayesian/

│   └── 1darcy/

├── sensitivity/

│   └── 1darcy/

├── oed/

│   └── 1darcy/

└── figures/

```



The exact directory names may be adjusted by the production runners, but every analysis should remain traceable to its input run.



Generated results are excluded from the public source repository.



\---



\## 24. Publication Figures and Tables



Final publication figures should record:



```text

source analysis run

Git commit

dataset

parameter configuration

figure-generating script

date generated

```



Figures placed in:



```text

assets/

```



should be curated intentionally.



Large collections of automatically generated figures should remain outside the public source repository.



\---



\## 25. Regression Tests



The repository contains a deterministic test suite.



Run:



```bash

python -m pytest -q

```



The current baseline is:



```text

60 passed

```



The suite covers:



```text

data contract

forward model

tensor rotation

Bayesian likelihood

Fisher information

geometry uncertainty

robust design

sensor placement

Bayesian design utilities

```



The normal test suite is intentionally fast and does not replace full production analyses.



\---



\## 26. Reproduction Levels



The project supports three levels of reproducibility.



\### Level 1 — software reproducibility



Verify that the current source tree functions:



```bash

python -m pytest -q

```



\### Level 2 — workflow reproducibility



Re-run the production scripts using the same:



```text

dataset

model configuration

parameters

seed

sample counts

```



\### Level 3 — scientific-result reproducibility



Reproduce the final figures, tables, posterior summaries, OED rankings, and EIG calculations using:



```text

same Git commit

same dataset

same environment

same numerical configuration

same convergence criteria

```



Level 3 is the standard expected for final reported results.



\---



\## 27. Recommended Run Record



For each final production analysis, maintain a simple record containing:



```text

Analysis name:

Date:

Git commit:

Dataset:

Runner:

Command:



Python:

NumPy:

pandas:

SciPy:

PyDREAM:

SALib:



Random seed:



Physical parameters:

Prior bounds:

Fixed inputs:



MCMC:

&#x20; chains:

&#x20; iterations:

&#x20; batch size:

&#x20; R-hat threshold:

&#x20; converged:



Sensitivity:

&#x20; OAT points:

&#x20; heatmap points:

&#x20; global samples:

&#x20; Sobol base:



OED:

&#x20; design type:

&#x20; design size:

&#x20; scenario count:

&#x20; metric:



Bayesian OED:

&#x20; posterior sample count:

&#x20; EIG outer:

&#x20; EIG inner:

&#x20; EIG standard error:



Output directory:

Notes:

```



This record can be stored alongside the final analysis results.



\---



\## 28. What Must Never Be Assumed



Do not assume that:



\- a program completing means an MCMC run converged;

\- a narrow posterior means a parameter is globally sensitive;

\- a large strain amplitude means a station is optimal;

\- a local Fisher optimum is globally optimal;

\- a prior-robust design remains optimal after Bayesian updating;

\- a small EIG Monte Carlo experiment gives a stable final ranking;

\- an inferred \\(\\sigma\_{\\mathrm{strain}}\\) represents instrument noise alone;

\- changing fixed model parameters leaves the analysis scientifically equivalent.



These distinctions are central to interpreting the AVANT workflow correctly.



\---



\## 29. Final Reproducibility Checklist



Before accepting a production result, verify:



```text

\[ ] Git commit recorded

\[ ] canonical dataset recorded

\[ ] station metadata recorded

\[ ] fixed model inputs recorded

\[ ] parameter names recorded

\[ ] prior bounds recorded

\[ ] random seed recorded

\[ ] software environment recorded

\[ ] execution command recorded

\[ ] MCMC convergence verified

\[ ] posterior metadata preserved

\[ ] postprocessing command preserved

\[ ] sensitivity settings preserved

\[ ] OED settings preserved

\[ ] Bayesian-OED settings preserved where applicable

\[ ] EIG Monte Carlo error checked where applicable

\[ ] final figures/tables linked to source run

```



A result is considered reproducible only when its computational provenance is complete enough to reconstruct the analysis.

