================================================
cryoSBI - Simulation-based Inference for Cryo-EM
================================================

.. start-badges

.. list-table::
    :stub-columns: 1

    * - tests
      - | |githubactions|
        

.. |githubactions| image:: https://github.com/DSilva27/cryo_em_SBI/actions/workflows/python-package.yml/badge.svg?branch=main
    :alt: Testing Status
    :target: https://github.com/DSilva27/cryo_em_SBI/actions

Summary
-------
XXX

Installing
----------
To install the module you will have to dowload the repository and create a virtual environment with the required dependencies.
You can create an environment for example with conda using the following command:

.. code:: bash

    conda create -n cryoSBI python=3.10

After creating the virtual environment, you should install the required dependencies and the module.

Dependencies
------------

1. `Lampe <https://lampe.readthedocs.io/en/stable/>`_.
2. `SciPy <https://scipy.org/>`_.
3. `Numpy <https://numpy.org/>`_.
4. `PyTorch <https://pytorch.org/get-started/locally/>`_.
5. json
6. `mrcfile <https://pypi.org/project/mrcfile/>`_.

Download this repository
------------------------
.. code:: bash

    git clone https://github.com/flatironinstitute/cryoSBI.git

Navigate to the cloned repository and install the module
--------------------------------------------------------
.. code:: bash
    
    cd cryoSBI

.. code:: bash

    pip install .

Generate model file to simulate cryo-EM particles
-------------------------------------------------
.. code:: bash

    make_torch_models \
        --pdb_files path_to_pdb_1.pdb path_to_pdb_2.pdb ... \
        --save_path path_to_save_models.pt \
        --atom_selection "name CA"

Training an amortized posterior model
--------------------------------------
.. code:: bash

    train_classifier \
        --image_config_file path_to_simulation_config_file.json \
        --train_config_file path_to_train_config_file.json\
        --epochs 150 \
        --estimator_file posterior.estimator \
        --loss_file posterior.loss \
        --n_workers 4 \
        --simulation_batch_size 5120 \
        --train_device cuda


Inference on experimental cryo-EM particles
-------------------------------------------
.. code:: bash
    classifier_inference \
        --folder_with_mrcs path_to_folder_with_mrc_files \
        --estimator_config path_to_estimator_config_file.json \
        --estimator_weights path_to_estimator_weights \
        --output_dir path_to_save_inference_results \
        --file_name inference_results.pt \
        --max_batch_size 256 \
        --num_workers 4 \
        --image_size 256 \
        --prefetch_factor 2

