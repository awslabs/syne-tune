# Change Log
All notable changes to this project will be documented in this file.
 
The format is based on [Keep a Changelog](http://keepachangelog.com/).

{{ $cutOffVersion := "0.4.2" }}
{{ $semverPattern := "^v\\d+\\.\\d+\\.\\d+$" }}

{{ if .Versions -}}
<a name="unreleased"></a>
## [Unreleased]

{{ if .Unreleased.CommitGroups -}}
{{ range .Unreleased.CommitGroups -}}
### {{ .Title }}
{{ range .Commits -}}
- {{ if .Scope }}**{{ .Scope }}:** {{ end }}{{ .Subject }}
{{ end }}
{{ end -}}
{{ end -}}
{{ end -}}

{{ range .Versions }}
{{ if and (ne "paper-experiments" .Tag.Name) (semverCompare .Tag.Name $cutOffVersion) }}
<a name="{{ .Tag.Name }}"></a>
## {{ if .Tag.Previous }}[{{ .Tag.Name }}]{{ else }}{{ .Tag.Name }}{{ end }} - {{ datetime "2006-01-02" .Tag.Date }}
{{ range .CommitGroups -}}
### {{ .Title }}
{{ range .Commits -}}
- {{ if .Scope }}**{{ .Scope }}:** {{ end }}{{ .Subject }}
{{ end }}
{{ end -}}

{{- if .RevertCommits -}}
### Reverts
{{ range .RevertCommits -}}
- {{ .Revert.Header }}
{{ end }}
{{ end -}}

{{- if .NoteGroups -}}
{{ range .NoteGroups -}}
### {{ .Title }}
{{ range .Notes }}
{{ .Body }}
{{ end }}
{{ end -}}
{{ end -}}
{{ end -}}

{{ end }}

{{- if .Versions }}
[Unreleased]: {{ .Info.RepositoryURL }}/compare/{{ $latest := index .Versions 0 }}{{ $latest.Tag.Name }}...HEAD
{{ range .Versions -}}
{{ if .Tag.Previous -}}
[{{ .Tag.Name }}]: {{ $.Info.RepositoryURL }}/compare/{{ .Tag.Previous.Name }}...{{ .Tag.Name }}
{{ end -}}
{{ end -}}
{{ end -}}





## [0.4.1] - 2023-03-16

We release version 0.4.1 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors:
@mseeger, @wesk, @sighellan, @aaronkl, @wistuba, @jgolebiowski, @610v4nn1, @geoalgo


### Added
* New tutorial: Using Syne Tune for Transfer Learning
* Multi-objective regularized evolution (MO-REA)
* Gaussian-process based methods support Box-Cox target transform and input
  warping
* Configuration can be passed as JSON file to evaluation script
* Remote tuning: Metrics are published to CloudWatch console

### Changed
* Refactoring and cleanup of `benchmarking`
* FIFOScheduler supports multi-objective schedulers
* Remote tuning with local backend: Checkpoints are not synced to S3
* Code in documentation is now tested automatically
* Benchmarking: Use longer descriptive names for result directories

### Fixed
* Specific state converter for DyHPO (can work poorly with the state converter
  for MOBSTER or Hyper-Tune)
* Fix of BOHB and SyncBOHB
* _objective_function parsing for YAHPO blackbox


## [0.4.0] - 2023-02-21

We release version 0.4.0 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors:
@mseeger, @wesk, @sighellan, @ondrejbohdal, @aaronkl, @wistuba, @jacekgo, @geoalgo


### Added
* New HPO algorithm: DyHPO
* New tutorial: Progressive ASHA
* Extended developer tutorial: Wrapping external scheduler code
* Schedulers can be restricted to set of configurations as subset of the
  full configuration space (option `restrict_configurations`)
* Data filtering for model-based multi-fidelity schedulers to avoid slowdown
  (option `max_size_data_for_model`)
* Input warping for Gaussian process covariance kernels
* Allow searchers to return the same configuration more than once (option 
  `allow_duplicates`). Unify duplicate filtering across all searchers
* Support `Ordinal` and `OrdinalNearestNeighbor` in `active_config_space` and
  warmstarting

### Changed
* Major simplifications in `benchmarking`
* Examples and documentation are encouraging to use `max_resource_attr`
  instead of `max_t`
* Major refactoring and extensions in testing and CI pipeline
* `RemoteLauncher` does not require custom container anymore

### Fixed
* `TensorboardCallback`: Logging of hyperparameters made optional, and updated
  example
* Refactoring of `BoTorchSearcher`
* Default value for DEHB baseline fixed
* Number of GPUs is now determined correctly


## [0.3.4] - 2023-01-11

We release version 0.3.4 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors!

### Changed
* Different searchers suggest the same initial random configurations if run
  with the same random seed
* Streamlined how SageMaker backend uses pre-built containers
* Improvements in documentation
* Improved random seed management in benchmarking
* New baseline wrappers: BOHB, ASHABORE, BOTorch, KDE

### Fixed
* Switch off hash checking of blackbox repository by default, since hash
  computation seem system-dependent. Hash computation will be fixed in a
  forthcoming release, and will be switched on again
* Fixed defaults of Hyper-Tune in benchmarking
* Bug fix of SyncMOBSTER (along with refactoring)
* Bug fix in REA (RegularizedEvolution)
* Bug fix in examples for constrained and cost-aware BO


## [0.3.3] - 2022-12-19

We release version 0.3.3 which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors (sorted by chronological commit order):
@mseeger, @mina-ghashami, @aaronkl, @jgolebiowski, @Valavanca, @TrellixVulnTeam,
@geoalgo, @wistuba, @mlblack

### Added
* Revamped documentation hosted at https://syne-tune.readthedocs.io
* New tutorial: Benchmarking in Syne Tune
* Added section on backends in Basics of Syne Tune tutorial
* Control of re-creating of blackboxes by checking and storing hash codes
* New benchmark: Transformer on WikiText-2
* Support SageMaker managed warm pools in SageMaker backend
* Improvements for benchmarking with YAHPO blackboxes
* Support points_to_evaluate in BORE
* SageMaker backend supports delete_checkpoints=True

### Changed
* GridSearch supports all domain types now
* BlackboxSurrogate of blackbox repository supports different modes
* Add timeout to unit tests
* New unit tests which run schedulers for longer, using simulator backend

### Fixed
* HyperbandScheduler: does_pause_resume for all types
* ASHA with type="promotion" did not work when checkpointing not implemented
* Fixed preprocessing of PD1 blackbox
* SageMaker backend reports on true number of busy workers (fixes issue #250)
* Fix issue with uploading/syncing to S3 of YAHPO blackbox
* Fix YAHPO surrogate evaluations in the presence of inactive hyperparameters
* Fix treatment of Status.paused in TuningStatus and Tuner


## [0.3.2] - 2022-10-14

We release 0.3.2 version which you can install with `pip install syne-tune[extra]`.

Thanks to all contributors (sorted by chronological commit order):
@mseeger, @geoalgo, @aaronkl, @wistuba, @talesa, @hfurkanbozkurt, @rsnirwan,
@duck105, @ondrejbohdal, @iaroslav-ai, @austinmw, @banyikun, @jjaeyeon, @ltiao

### Added
* New tutorial: How to Contribute a New Scheduler
* New tutorial: Multi-Fidelity Hyperparameter Optimization
* YAHPO benchmarks integrated into blackbox repository
* PD1 benchmarks integrated into blackbox repository
* New HPO algorithm: Hyper-Tune
* New HPO algorithm: Differential Evolution Hyperband (DEHB)
* New experimental HPO algorithm: Neuralband
* New HPO algorithm: Grid search (categorical variables only)
* BOTorch searcher
* MOBSTER algorithm supports independent GPs at each rung level
* Support for launching experiments in benchmarking/commons, for local,
  SageMaker, and simulator backend
* New benchmark: Fine-tuning Hugging Face transformers
* Add IPython util function to display results as parallel categories plot
* New hyperparameter types `ordinal`, `logordinal`
* Support no checkpointing in BlackboxRepositoryBackend
* Plateau rule as StoppingCriterion
* Automate PyPI releases: python-publish.yml
* Add license hook

### Changed
* Replace PyTorch MLP by sklearn in BORE (better performance)
* AWS dependencies moved out of core into `aws`
* New dependencies `yahpo`
 
### Fixed
* In SageMaker backend, trials with low IDs received reports several times. This
  is fixed
* Fixing issue with checkpoint_s3_uri usage
* Fix mode in BOTorch searcher when maximizing
* Avoid experiment abort due to throttling of SageMaker job launching
* Surrogate model for lcbench defaults to 1-NN now
* Fix conditional imports, so Syne Tune can be run with reduced dependencies
* Fix lcbench blackbox (ignore first and last fidelity)
* Fix bug in BlackboxSimulatorBackend for pause/resume scheduling (issue #304)
* Revert wait_trial_completion_when_stopping to False
* Terminate with error when tuning sees an exception
* Docker Building Fixed by Adding Line Breaks At End of Requirements Files 
* Control Decision for Running Trials When Stopping Criterion is Met
* Fix mode MSR and HB+BB


## [0.3.0] - 2022-05-07

We release 0.3.0 version which you can install with `pip install syne-tune[extra]==0.3.0`, thanks to all contributors!
(sorted by chronological commit order :-))
@mseeger, @geoalgo, @iaroslav-ai, @aaronkl, @wistuba, @rsnirwan, @hfurkanbozkurt, @ondrejbohdal, @ltiao, @lostella, @jjaeyeon
 
### Added
* Tensorboard visualization 
* Option to change the path where experiments results are written
* Revamp of README.md, addition of an FAQ
* Code formatting with Black
* Scripts for launching and analysing benchmark results of Syne Tune AutoML paper
* Option to avoid suffixing the tuner name with a random hash
* Tutorial for PASHA
* Adds support to run BORE with Hyperband
* Updates python version to 3.8 in remote launcher as 3.6 is EOL
* Support custom images in remote launcher
* Add ZS/SMFO scheduler
* Add support to tune python functions directly rather than file scripts (experimental)  
* Now display where results are written at the end of the tuning in addition to the beginning 
* Log config space with tuning metadata
* Add option to delete checkpoints of finished trials and improve support of checkpointing for SageMaker backend
* Add support for adding surrogate on top of blackboxes evaluations
 
### Changed
* Removed deprecated `search_space` which has been renamed `config_space` in all the code base
* Removed some unused dependencies
 
### Fixed
* fix conditional imports
* avoid sampling twice the same candidate in REA
* print tuning statistics only for numerics
* allow to load blackboxes without s3 credentials
* fix sagemaker root path in remote launcher
* fix a typo in product kernel implementation