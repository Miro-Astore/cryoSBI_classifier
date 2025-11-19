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

Installing
----------
To install the module you will have to dowload the repository and create a virtual environment with the required dependencies.
You can create an environment for example with conda using the following command:

.. code:: bash

    conda create -n cryoSBI python=3.10

After creating the virtual environment, you should install the required dependencies and the module.

Dependencies
------------

#. `Zuko <https://pypi.org/project/zuko/>`_.
#. `PyTorch <https://pytorch.org/get-started/locally/>`_.
#. `NumPy <https://numpy.org/>`_.
#. `Matplotlib <https://matplotlib.org/>`_.
#. `SciPy <https://scipy.org/>`_.
#. `TorchVision <https://pytorch.org/vision/stable/>`_.
#. `mrcfile <https://pypi.org/project/mrcfile/>`_.
#. `tqdm <https://pypi.org/project/tqdm/>`_.

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
.. code-block:: bash

    make_torch_models \
        --pdb_files path_to_pdb_1.pdb path_to_pdb_2.pdb ... \
        --save_path path_to_save_models.pt \
        --atom_selection "name CA"

Training classifier for amortized inference
-------------------------------------
.. code-block:: bash

    train_classifier \
        --image_config_file path_to_simulation_config_file.json \
        --train_config_file path_to_train_config_file.json \
        --epochs 150 \
        --estimator_file path_to_estimator_file.pt \
        --loss_file posterior.loss \
        --n_workers 4 \
        --simulation_batch_size 5120 \
        --train_device cuda

Inference on cryo-EM particles
------------------------------
.. code-block:: bash

    classifier_inference \
        --folder_with_mrcs path_to_folder_with_mrc_files \
        --estimator_config path_to_estimator_config_file.json \
        --estimator_weights path_to_estimator_weights.pt \
        --output_dir path_to_save_inference_results \
        --file_name inference_results.pt \
        --max_batch_size 256 \
        --num_workers 4 \
        --image_size 256 \
        --prefetch_factor 2

