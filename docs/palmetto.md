\# Palmetto HPC Deployment



\## 1. Purpose



This document describes deployment and execution of the AVANT production workflow on Clemson University's Palmetto HPC cluster.



The instructions intentionally use



```text

$HOME

```



rather than a specific username or personal home-directory path.



The currently verified Palmetto Python access path is the Anaconda module



```text

anaconda3/2023.09-0

```



available through Palmetto's module system.



The exact Python package versions, SLURM resource requirements, and final production environment should be recorded after the complete AVANT workflow has been validated on Palmetto.



\---



\## 2. Production Philosophy



The Palmetto deployment should reproduce the same scientific workflow already validated locally.



Moving from a workstation to HPC changes:



```text

compute resources

parallel execution

walltime

storage

job scheduling

```



but should not silently change:



```text

dataset

parameter definitions

prior bounds

fixed analytical inputs

forward-model equations

likelihood

convergence criterion

sensitivity definitions

OED definitions

scientific interpretation

```



Any change to those scientific assumptions represents a new analysis configuration and should be documented.



\---



\## 3. Recommended Repository Location



A convenient location is



```text

$HOME/Avant\_multi\_station

```



\### First-time clone



From the Palmetto login node:



```bash

cd "$HOME"



git clone git@github.com:PapaKwansa/Avant\_multi\_station.git



cd Avant\_multi\_station

```



Verify:



```bash

git status

git branch --show-current

git rev-parse HEAD

```



\### Existing checkout



If the repository already exists:



```bash

cd "$HOME/Avant\_multi\_station"



git status

git pull

```



Before a production run, always record:



```bash

git rev-parse HEAD

git status --short

```



Prefer a clean working tree for final production calculations.



\---



\## 4. Load Python on Palmetto



Palmetto provides Anaconda through the module system.



Inspect the available module with:



```bash

module spider anaconda

```



The currently verified module is:



```text

anaconda3/2023.09-0

```



Load it with:



```bash

module load anaconda3/2023.09-0

```



Verify:



```bash

python --version

which python

conda --version

```



Also record the loaded modules:



```bash

module list

```



The Anaconda module must be loaded in each new shell or SLURM job before activating the AVANT Conda environment.



\---



\## 5. Create a Dedicated AVANT Conda Environment



Do not install the AVANT dependencies directly into Palmetto's shared/base Anaconda environment.



After loading Anaconda:



```bash

module load anaconda3/2023.09-0

```



create a dedicated environment:



```bash

conda create -n avant python=3.9

```



Activate it:



```bash

conda activate avant

```



Verify:



```bash

python --version

which python

```



The environment name `avant` is recommended for clarity but is not scientifically significant.



\---



\## 6. Install the Repository



Enter the repository:



```bash

cd "$HOME/Avant\_multi\_station"

```



Install the core package:



```bash

python -m pip install -e .

```



For Bayesian inversion:



```bash

python -m pip install -e ".\[inversion]"

```



For sensitivity analysis:



```bash

python -m pip install -e ".\[sensitivity]"

```



For plotting:



```bash

python -m pip install -e ".\[plotting]"

```



For testing:



```bash

python -m pip install -e ".\[test]"

```



For the complete AVANT analysis environment:



```bash

python -m pip install -e ".\[inversion,sensitivity,plotting,test]"

```



Verify the package:



```bash

python -c "import avant\_model; print('avant\_model: OK')"

```



\---



\## 7. Record the Environment



After the environment has been installed successfully, record:



```bash

python --version

which python

python -m pip list

module list

```



A Conda environment specification can also be preserved:



```bash

conda env export > avant\_environment.yml

```



For final production analyses, retain the environment specification with the analysis provenance.



Do not assume that a Python environment created for another research project is automatically suitable for AVANT.



\---



\## 8. Validate the Installation



Before submitting an expensive calculation, run the complete deterministic regression suite:



```bash

cd "$HOME/Avant\_multi\_station"



python -m pytest -q

```



The current release baseline is:



```text

60 passed

```



A failure on Palmetto that does not occur locally should be investigated before production calculations are launched.



Also verify the production command-line interfaces:



```bash

python scripts/preprocessing/clean\_dataset.py --help



python scripts/inversion/run\_inversion\_multi\_station.py --help



python scripts/postprocessing/postprocess\_bayesian\_1darcy.py --help



python scripts/sensitivity/run\_sensitivity\_1darcy.py --help



python scripts/oed/run\_oed\_1darcy.py --help



python scripts/oed/run\_bayesian\_oed\_1darcy.py --help

```



\---



\## 9. Verify the Canonical Dataset



The production 1-Darcy data should exist under:



```text

datasets/comsol/1darcy/

```



Check:



```bash

ls -l datasets/comsol/1darcy/metadata/

ls -l datasets/comsol/1darcy/processed/

ls -l datasets/comsol/1darcy/raw/

```



The canonical processed files are:



```text

datasets/comsol/1darcy/metadata/stations.csv

datasets/comsol/1darcy/processed/strain.csv

```



The expected processed-data contract is:



```text

8 stations

107 time steps

4 strain components per station

32 strain channels

3424 strain observations

```



The regression suite verifies this contract.



\---



\## 10. First Palmetto Scientific Smoke Test



After the 60-test suite passes, run the inexpensive local Fisher OED calculation:



```bash

python scripts/oed/run\_oed\_1darcy.py --mode local

```



For the current canonical reference configuration, the validated local result is approximately:



```text

rank = 6/6

condition number = 219.371

log det(F) = 101.974

```



This is a useful cross-platform scientific regression check in addition to the unit tests.



Material differences should be investigated before launching production jobs.



\---



\## 11. Bayesian Smoke Test



A small Bayesian run can verify:



\- PyDREAM installation;

\- multiprocessing;

\- likelihood execution;

\- analytical forward-model evaluation;

\- output writing;

\- run-manifest generation;

\- convergence bookkeeping.



For example:



```bash

python scripts/inversion/run\_inversion\_multi\_station.py \\

&#x20;   --max-iter 20 \\

&#x20;   --batch-size 20 \\

&#x20;   --nchains 4 \\

&#x20;   --seed 42

```



This is a smoke test only.



It is expected that such a short run may finish with:



```text

not\_converged

```



That can still represent a successful software validation.



It must not be treated as a final scientific posterior.



\---



\## 12. Login Nodes Versus Compute Nodes



Login nodes should be used for lightweight tasks such as:



```text

repository management

environment setup

small import checks

pytest

short CLI validation

small deterministic smoke tests

job submission

result inspection

```



Long Bayesian inversion, large Sobol analyses, robust OED ensembles, placement searches, and production EIG calculations should run through SLURM compute jobs.



Do not use the login node for long production calculations.



\---



\## 13. Benchmark Before Choosing SLURM Resources



Before selecting production CPU, memory, and walltime requests:



1\. run the 60-test suite;

2\. run the local Fisher OED check;

3\. benchmark a representative forward-model calculation;

4\. benchmark a small PyDREAM batch;

5\. inspect CPU utilization;

6\. inspect memory usage;

7\. estimate production runtime.



The final resource requests should be based on measured AVANT behavior on Palmetto rather than assumptions from another application.



\---



\## 14. Recommended Output Organization



The source checkout can remain at:



```text

$HOME/Avant\_multi\_station

```



During initial deployment, generated outputs can use the repository's existing hierarchy:



```text

results/

├── bayesian/

│   └── 1darcy/

├── sensitivity/

│   └── 1darcy/

└── oed/

&#x20;   └── 1darcy/

```



Logs can be written to:



```text

logs/

```



Both `results/` and `logs/` are excluded from Git.



If production output becomes too large for the home filesystem, move generated results to an appropriate Palmetto project or scratch filesystem and record the exact location in the run metadata.



Do not mix AVANT production outputs with unrelated PFLOTRAN or other project results.



\---



\## 15. Create the Log Directory



Before submitting SLURM jobs:



```bash

cd "$HOME/Avant\_multi\_station"



mkdir -p logs

```



This allows SLURM output files to use:



```text

logs/<job-name>\_<job-id>.out

logs/<job-name>\_<job-id>.err

```



\---



\## 16. Activating AVANT Inside SLURM



A SLURM batch shell does not automatically inherit every interactive-shell setup assumption.



A production job should explicitly load Anaconda and activate the AVANT environment.



Use:



```bash

module load anaconda3/2023.09-0

```



then initialize/activate Conda using the method verified on Palmetto.



A typical pattern is:



```bash

module load anaconda3/2023.09-0



source "$(conda info --base)/etc/profile.d/conda.sh"



conda activate avant

```



Verify inside the job:



```bash

python --version

which python

```



This prevents a SLURM job from accidentally using the wrong Python interpreter.



\---



\## 17. Generic Bayesian SLURM Script



After benchmarking determines appropriate resources, a Bayesian production script can follow this pattern:



```bash

\#!/bin/bash



\#SBATCH --job-name=avant\_bayes

\#SBATCH --output=logs/avant\_bayes\_%j.out

\#SBATCH --error=logs/avant\_bayes\_%j.err



\# Add tested Palmetto resource requests here, for example:

\#

\# #SBATCH --nodes=1

\# #SBATCH --ntasks=1

\# #SBATCH --cpus-per-task=<N>

\# #SBATCH --mem=<MEMORY>

\# #SBATCH --time=<HH:MM:SS>

\# #SBATCH --partition=<PARTITION>



set -euo pipefail



cd "$HOME/Avant\_multi\_station"



module load anaconda3/2023.09-0



source "$(conda info --base)/etc/profile.d/conda.sh"



conda activate avant



echo "============================================================"

echo "AVANT BAYESIAN PRODUCTION RUN"

echo "============================================================"



echo "Date:"

date



echo "Hostname:"

hostname



echo "SLURM job ID:"

echo "${SLURM\_JOB\_ID:-not\_set}"



echo "Git commit:"

git rev-parse HEAD



echo "Git status:"

git status --short



echo "Python:"

python --version



echo "Python executable:"

which python



echo "Loaded modules:"

module list 2>\&1



python scripts/inversion/run\_inversion\_multi\_station.py \\

&#x20;   --model-name pydream\_1darcy\_production \\

&#x20;   --seed 42

```



Do not copy arbitrary CPU/memory/walltime values into the final production script before benchmarking.



\---



\## 18. Why PyDREAM Chain Count Matters for Resources



The production inversion uses multiple PyDREAM chains.



The number of chains is controlled by:



```text

\--nchains

```



The sampler can evaluate chains through multiprocessing.



Therefore the requested CPU allocation should be chosen consistently with the intended chain configuration.



A large CPU request does not automatically make the inversion faster if the sampler cannot use those CPUs.



Benchmark the intended number of chains before finalizing:



```text

\--cpus-per-task

```



and other SLURM resource settings.



\---



\## 19. Production Bayesian Inversion



Inspect the current interface before submission:



```bash

python scripts/inversion/run\_inversion\_multi\_station.py --help

```



A production run should use:



\- multiple chains;

\- sufficient maximum iterations;

\- an appropriate batch size;

\- the production R-hat threshold;

\- a reproducible seed;

\- a unique model name.



Do not simply reuse the 20-iteration smoke-test settings.



The final production command should be recorded in the job log or run provenance.



\---



\## 20. Bayesian Output Directory



The default production hierarchy is:



```text

results/bayesian/1darcy/<model-name>/

```



For example:



```text

results/bayesian/1darcy/pydream\_1darcy\_production/

```



Important outputs include metadata such as:



```text

run\_manifest.json

posterior\_run\_summary.json

convergence history

chain histories

log-probability histories

```



Do not overwrite a previous scientifically important run without intentionally archiving it.



\---



\## 21. Convergence Verification



A completed SLURM job is not automatically a converged Bayesian inversion.



After the job finishes, inspect:



```text

posterior\_run\_summary.json

```



and the associated convergence outputs.



The production criterion requires the configured R-hat threshold to be satisfied for all sampled parameters.



The current default threshold is:



\\\[

\\hat R < 1.1.

\\]



The critical distinction is:



```text

run\_status = converged

```



versus:



```text

run\_status = not\_converged

```



Only a converged posterior should normally be used for final scientific interpretation or posterior-informed OED.



\---



\## 22. Validate the Bayesian Run Contract



Before posterior-dependent analyses:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode validate

```



This verifies the Bayesian-run contract and reports its convergence state.



Do not use:



```text

\--allow-nonconverged

```



for production scientific analyses.



That option is intended for software testing and diagnostics.



\---



\## 23. Posterior Postprocessing



After convergence has been verified:



```bash

python scripts/postprocessing/postprocess\_bayesian\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name>

```



Inspect the available controls with:



```bash

python scripts/postprocessing/postprocess\_bayesian\_1darcy.py --help

```



The postprocessor generates:



```text

trace diagnostics

posterior summaries

posterior distributions

credible intervals

parameter correlations

pairwise posterior relationships

observed-versus-predicted strain

parameter-only predictive uncertainty

total predictive uncertainty

residual diagnostics

posterior predictive metrics

```



Preserve the postprocessing command with the production results.



\---



\## 24. Sensitivity Analysis on Palmetto



Inspect:



```bash

python scripts/sensitivity/run\_sensitivity\_1darcy.py --help

```



A lightweight validation configuration is:



```bash

python scripts/sensitivity/run\_sensitivity\_1darcy.py \\

&#x20;   --sobol-base 32 \\

&#x20;   --oat-points 7 \\

&#x20;   --heatmap-points 9 \\

&#x20;   --global-samples 100

```



These settings are for software validation and exploratory analysis.



Production sensitivity analysis should use larger sample counts and verify stability of the resulting conclusions.



Large OAT, response-surface, global-sampling, and Sobol calculations should be submitted through SLURM.



\---



\## 25. Classical OED on Palmetto



The production runner is:



```text

scripts/oed/run\_oed\_1darcy.py

```



Available analysis modes include:



```text

local

existing

robust

boundary

placement

```



Inspect the exact interface:



```bash

python scripts/oed/run\_oed\_1darcy.py --help

```



The inexpensive local calculation is useful for validation:



```bash

python scripts/oed/run\_oed\_1darcy.py --mode local

```



Larger robust geometry ensembles and placement searches should be submitted through SLURM.



\---



\## 26. Bayesian OED on Palmetto



Bayesian OED depends on a Bayesian posterior.



Validate the run first:



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode validate

```



\### Existing-network posterior OED



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode posterior\_existing

```



\### Posterior-informed spatial placement



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode posterior\_placement

```



\### Expected Information Gain



```bash

python scripts/oed/run\_bayesian\_oed\_1darcy.py \\

&#x20;   --run-dir results/bayesian/1darcy/<model-name> \\

&#x20;   --mode eig

```



Production posterior OED and EIG should use a converged posterior.



\---



\## 27. EIG Resource Considerations



Expected Information Gain is more computationally expensive than local Fisher OED because it uses nested Monte Carlo calculations.



Important controls include:



```text

\--posterior-samples

\--eig-designs

\--eig-outer

\--eig-inner

```



Small values such as those used during local software validation are not intended for final scientific ranking.



Production EIG should:



1\. use sufficiently many posterior scenarios;

2\. use larger outer and inner Monte Carlo counts;

3\. record the random seed;

4\. report the estimated Monte Carlo standard error;

5\. check ranking stability as sample counts increase.



A production EIG calculation is a good candidate for an independent SLURM job.



\---



\## 28. Job Dependencies



Some stages are independent while others depend scientifically on previous results.



A useful conceptual workflow is:



```text

&#x20;                  Bayesian inversion

&#x20;                         |

&#x20;                         v

&#x20;                convergence check

&#x20;                         |

&#x20;                         v

&#x20;                posterior processing

&#x20;                         |

&#x20;                         v

&#x20;             posterior-dependent OED

&#x20;                         |

&#x20;                         v

&#x20;                        EIG

```



Classical sensitivity analysis and prior-based classical OED can be run independently of posterior convergence.



A more complete workflow is therefore:



```text

&#x20;                     canonical data

&#x20;                          |

&#x20;            +-------------+-------------+

&#x20;            |                           |

&#x20;            v                           v

&#x20;     Bayesian inversion        sensitivity / prior OED

&#x20;            |

&#x20;            v

&#x20;     convergence check

&#x20;            |

&#x20;            v

&#x20;      postprocessing

&#x20;            |

&#x20;            v

&#x20;     Bayesian OED / EIG

```



Do not automatically launch posterior-dependent jobs merely because the inversion SLURM job exited successfully.



The convergence state must be checked first.



\---



\## 29. SLURM Dependencies



After the workflow is validated, SLURM dependencies may be used to automate safe stages.



For example:



```bash

JOB1=$(sbatch avant\_bayesian.slurm | awk '{print $4}')

```



A subsequent job can depend on successful completion with:



```bash

sbatch --dependency=afterok:$JOB1 next\_job.slurm

```



However, `afterok` verifies successful process termination, not Bayesian convergence.



Therefore posterior-dependent analyses should still include an explicit convergence check.



\---



\## 30. Production Logging



Every production SLURM job should preserve:



```text

SLURM job ID

job name

stdout

stderr

start time

end time

hostname

Git commit

Git working-tree state

Python version

Python executable

loaded modules

Conda environment

execution command

exit status

```



Use the SLURM job ID in log names.



For example:



```text

logs/avant\_bayes\_<jobid>.out

logs/avant\_bayes\_<jobid>.err

```



\---



\## 31. Environment Recording



For a production run, record:



```bash

module list

python --version

which python

python -m pip list

```



Optionally preserve:



```bash

conda env export > avant\_environment.yml

```



The environment specification should be associated with the corresponding production analysis.



\---



\## 32. Updating the Palmetto Checkout



Before pulling:



```bash

cd "$HOME/Avant\_multi\_station"



git status

```



Then:



```bash

git pull

```



Verify:



```bash

git rev-parse HEAD

git status --short

```



For final production calculations, prefer a clean checkout corresponding to a known Git commit.



\---



\## 33. First-Time Palmetto Deployment Checklist



```text

\[ ] clone/update repository



\[ ] module spider anaconda



\[ ] module load anaconda3/2023.09-0



\[ ] verify python, conda, and module list



\[ ] create dedicated AVANT Conda environment



\[ ] activate AVANT environment



\[ ] install AVANT package and optional dependencies



\[ ] import avant\_model successfully



\[ ] run 60-test regression suite



\[ ] verify canonical 1-Darcy dataset



\[ ] run local Fisher OED regression



\[ ] run short Bayesian smoke test



\[ ] verify smoke-run output contract



\[ ] benchmark representative forward-model runtime



\[ ] benchmark a small PyDREAM batch



\[ ] determine SLURM CPU/memory/walltime requirements



\[ ] submit production Bayesian inversion



\[ ] verify convergence



\[ ] run posterior postprocessing



\[ ] run production sensitivity analysis



\[ ] run classical/robust OED



\[ ] run posterior-informed Bayesian OED



\[ ] run production EIG



\[ ] preserve provenance and final results

```



\---



\## 34. Production Environment Record



After the first complete validated Palmetto deployment, update this section with the tested environment.



```text

Palmetto Anaconda module:

&#x20;   anaconda3/2023.09-0



Python version:

&#x20;   TO BE VERIFIED



Environment manager:

&#x20;   Conda



Environment name:

&#x20;   avant



NumPy:

&#x20;   TO BE VERIFIED



pandas:

&#x20;   TO BE VERIFIED



SciPy:

&#x20;   TO BE VERIFIED



Matplotlib:

&#x20;   TO BE VERIFIED



PyDREAM:

&#x20;   TO BE VERIFIED



SALib:

&#x20;   TO BE VERIFIED



pytest:

&#x20;   TO BE VERIFIED



Palmetto partition:

&#x20;   TO BE BENCHMARKED



Nodes:

&#x20;   TO BE BENCHMARKED



CPUs:

&#x20;   TO BE BENCHMARKED



Memory:

&#x20;   TO BE BENCHMARKED



Walltime:

&#x20;   TO BE BENCHMARKED

```



Do not replace the `TO BE VERIFIED` or `TO BE BENCHMARKED` entries by assumption.



Update them only after the corresponding Palmetto tests have actually been run.



\---



\## 35. Reproducibility Record for Each Palmetto Run



Each final production run should retain:



```text

Analysis name:

Date:

SLURM job ID:



Git commit:

Git branch:

Git working tree clean:



Anaconda module:

Conda environment:

Python version:



Dataset:

Station metadata:



Parameterization:

Prior bounds:

Fixed analytical inputs:



Random seed:



SLURM:

&#x20;   partition:

&#x20;   nodes:

&#x20;   tasks:

&#x20;   CPUs:

&#x20;   memory:

&#x20;   walltime:



Bayesian:

&#x20;   chains:

&#x20;   max iterations:

&#x20;   batch size:

&#x20;   R-hat threshold:

&#x20;   converged:



Sensitivity:

&#x20;   OAT points:

&#x20;   heatmap points:

&#x20;   global samples:

&#x20;   Sobol base:



OED:

&#x20;   mode:

&#x20;   design size:

&#x20;   scenarios:

&#x20;   metric:



Bayesian OED:

&#x20;   posterior samples:

&#x20;   posterior source:

&#x20;   posterior converged:



EIG:

&#x20;   designs:

&#x20;   outer samples:

&#x20;   inner samples:

&#x20;   standard error:



Output directory:

Notes:

```



\---



\## 36. Recommended Production Sequence



The recommended final Palmetto workflow is:



```text

Clone/update repository

&#x20;         |

&#x20;         v

Load anaconda3/2023.09-0

&#x20;         |

&#x20;         v

Activate dedicated AVANT environment

&#x20;         |

&#x20;         v

Run 60 regression tests

&#x20;         |

&#x20;         v

Verify canonical dataset

&#x20;         |

&#x20;         v

Run local Fisher regression

&#x20;         |

&#x20;         v

Benchmark AVANT

&#x20;         |

&#x20;         +-----------------------+

&#x20;         |                       |

&#x20;         v                       v

&#x20;Bayesian inversion       Sensitivity / prior OED

&#x20;         |

&#x20;         v

&#x20;Verify convergence

&#x20;         |

&#x20;         v

&#x20;Posterior postprocessing

&#x20;         |

&#x20;         v

&#x20;Posterior Bayesian OED

&#x20;         |

&#x20;         v

&#x20;Production EIG

&#x20;         |

&#x20;         v

&#x20;Preserve provenance

```



\---



\## 37. What Should Not Be Hard-Coded



The public repository documentation should not hard-code:



```text

personal username

/home/<specific-user>

temporary scratch directory

untested SLURM partition

untested CPU allocation

untested memory request

untested walltime

```



Use:



```text

$HOME

```



for portable home-directory examples.



Resource settings should be filled in only after benchmarking on the target Palmetto environment.



\---



\## 38. Important Scientific Distinction



HPC execution improves computational capacity.



It does not change the standard for scientific acceptance.



A production result still requires:



```text

correct dataset

correct model configuration

documented priors

documented fixed inputs

reproducible seed

successful execution

appropriate convergence

stable numerical analysis

complete provenance

```



In particular:



> A successful SLURM job does not imply a converged Bayesian posterior.



and:



> A fast low-sample EIG run does not imply a stable production EIG ranking.



\---



\## 39. Related Documentation



See:



```text

README.md



docs/workflow.md



docs/bayesian\_inversion.md



docs/sensitivity\_analysis.md



docs/optimal\_experimental\_design.md



docs/reproducibility.md

```



for the complete scientific and computational workflow.

