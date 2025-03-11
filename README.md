# Why do Machine Learning Notebooks Crash?

This is the official repository for our paper "Why do Machine Learning Notebooks Crash?", which is currently under review.
It is an empirical study on software crashes in ML programs written in Jupyter notebooks.

Paper link: https://arxiv.org/abs/2411.16795


## Expirimental Setup:

We collect notebooks from two sources:
* GitHub:
    + Download from the deduplicated version of [The Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup/tree/main/data/jupyter-notebook)[1] dataset. 
    + The downloaded files are .parquet files that can be transformed into .ipynb files like [this](./nbdata_github_thestack/parquet_to_nbs.ipynb).
    + GitHub notebooks used in this study is between 2015.1.1 and 2022.3.31.
* Kaggle:
    + Query and download via [KGTorrent](https://github.com/collab-uniba/KGTorrent) [2] with the [Meta Kaggle dataset](https://www.kaggle.com/datasets/kaggle/meta-kaggle). The Meta Kaggle dataset we use is from Feb. 2024.
    + Kaggle notebooks used in this paper is from 2023.1.1 to 2024.1.1.

We then filter the dataset based on:
* Notebooks with error outputs:
    + [data_filtering_[filtering1]errors.ipynb](./data_filtering_[filtering1]errors.ipynb)
* Notebooks that are valid Python3 ML notebooks with relevent exception types:
    + [data_filtering_[process]builtin_exceptions.ipynb](./data_filtering_[process]builtin_exceptions.ipynb)
    + [data_filtering_[process]evalues.ipynb](./data_filtering_[process]evalues.ipynb)
    + [data_filtering_[process]traceback.ipynb](./data_filtering_[process]traceback.ipynb)
    + [data_filtering_[process]valid_ast.ipynb](./data_filtering_[process]valid_ast.ipynb)
    + [data_filtering_[filtering2]ML_exp_valid.ipynb](./data_filtering_[filtering2]ML_exp_valid.ipynb)

* The ML libraries used to identify ML notebooks are explained in:
    + [data_filtering_[analysis]ML_library.ipynb](./data_filtering_[analysis]ML_library.ipynb)

* We identify which programming language GitHub notebooks are written in in:
    + [data_filtering_[analysis]programminglanguage.ipynb](./data_filtering_[analysis]programminglanguage.ipynb)

The sampling procedure is illustrated here:
* How the sample sizes are calculated:
    + [data_sampling_[sampling]_size_sample.ipynb](./data_sampling_[sampling]_size_sample.ipynb)
* How the data is clustered and sampled:
    + [data_sampling_[clustering]jaccard_similarity.ipynb](./data_sampling_[clustering]jaccard_similarity.ipynb)
* How the resampling is done:
    + [data_sampling_[resampling].ipynb](./data_sampling_[resampling].ipynb)

The results, from manual labeling and reviewing on the sampled dataset, can be found here:
* Data process:
    + [data_analysis_[0_process]_dataprepare.ipynb](./data_analysis_[0_process]_dataprepare.ipynb)
* Plots for RQs:
    + [data_analysis_[plots]_rqs.ipynb](./data_analysis_[plots]_rqs.ipynb)
    + [data_analysis_[plots]_rq2_libraries.ipynb](./data_analysis_[plots]_rq2_libraries.ipynb)
* Statistic tests:
    + [data_analysis_[statistictests].ipynb](./data_analysis_[statistictests].ipynb)
* Comparison with existing studies:
    + [data_analysis_[comparison]_existing_studies.ipynb](./data_analysis_[comparison]_existing_studies.ipynb)

## Shared Dataset
All the related data can be found on [zenodo](https://doi.org/10.5281/zenodo.14070488), including:
* GitHub and Kaggle notebooks that *contain error outputs*.
* Identified programming language results of GitHub notebooks.
* Identified ML library results from Kaggle notebooks.
* Datasets of crashes from GitHub and Kaggle.
* Clustering results of crashes from all crashes, and from GitHub and Kaggle respectively.
* Sampled crashes and associated notebooks.
* Manual labeling and reviewing results.
* Reproducing results.

* Note: the *full* GitHub notebook dataset is easily downloadable from [The Stack](https://huggingface.co/datasets/bigcode/the-stack-dedup/tree/main/data/jupyter-notebook), For the Kaggle notebooks, which have daily HTTP download limitations, we provide access upon request.

### References
[1] D. Kocetkov, R. Li, L. B. Allal, J. Li, C. Mou, C. M. Ferrandis, Y. Jernite, M. Mitchell, S. Hughes, T. Wolf, D. Bahdanau, L. von Werra, and H. de Vries, “The Stack: 3 TB of permissively licensed source code,” in arXiv:2211.15533, 2022.

[2] Quaranta, Luigi, Calefato, Fabio, & Lanubile, Filippo. (2021). collab-uniba/KGTorrent: First release (v. 1.0.0) of KGTorrent (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.4472990
