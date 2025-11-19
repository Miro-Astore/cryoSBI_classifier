import argparse
from cryo_sbi.utils.generate_models import make_torch_models
from cryo_sbi.inference.train import train_classifier
from cryo_sbi.inference.inference import classifier_inference


def cl_make_torch_models():
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--pdb_files", action="store", type=str, required=True)
    cl_parser.add_argument("--output_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--atom_selection", action="store", type=str, required=False, default="all"
    )
    args = cl_parser.parse_args()
    pdb_files = args.pdb_files.split(",")
    make_torch_models(
        pdb_files=pdb_files,
        save_path=args.output_file,
        atom_selection=args.atom_selection,
    )


def cl_train():
    cl_parser = argparse.ArgumentParser()

    cl_parser.add_argument(
        "--image_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument(
        "--train_config_file", action="store", type=str, required=True
    )
    cl_parser.add_argument("--epochs", action="store", type=int, required=True)
    cl_parser.add_argument("--estimator_file", action="store", type=str, required=True)
    cl_parser.add_argument("--loss_file", action="store", type=str, required=True)
    cl_parser.add_argument(
        "--train_from_checkpoint",
        action="store",
        type=bool,
        nargs="?",
        required=False,
        const=True,
        default=False,
    )
    cl_parser.add_argument(
        "--state_dict_file", action="store", type=str, required=False, default=False
    )
    cl_parser.add_argument(
        "--n_workers", action="store", type=int, required=False, default=1
    )
    cl_parser.add_argument(
        "--train_device", action="store", type=str, required=False, default="cpu"
    )
    cl_parser.add_argument(
        "--saving_freq", action="store", type=int, required=False, default=20
    )
    cl_parser.add_argument(
        "--simulation_batch_size",
        action="store",
        type=int,
        required=False,
        default=1024,
    )

    args = cl_parser.parse_args()

    train_classifier(
        image_config=args.image_config_file,
        train_config=args.train_config_file,
        epochs=args.epochs,
        estimator_file=args.estimator_file,
        loss_file=args.loss_file,
        train_from_checkpoint=args.train_from_checkpoint,
        model_state_dict=args.state_dict_file,
        n_workers=args.n_workers,
        device=args.train_device,
        saving_frequency=args.saving_freq,
        simulation_batch_size=args.simulation_batch_size,
    )


def cl_inference():
    cl_parser = argparse.ArgumentParser()
    cl_parser.add_argument("--folder_with_mrcs", type=str, required=True)
    cl_parser.add_argument("--estimator_weights", type=str, required=True)
    cl_parser.add_argument("--estimator_config", type=str, required=True)
    cl_parser.add_argument("--file_name", type=str, required=True)
    cl_parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading (default: 2)"
    )
    cl_parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save outputs (default: current directory)"
    )
    cl_parser.add_argument(
        "--image_size", type=int, default=128, help="Image size (default: 128)"
    )
    cl_parser.add_argument(
        "--prefetch_factor", type=int, default=2, help="Prefetch factor for data loading (default: 2)"
    )
    cl_parser.add_argument(
        "--max_batch_size", type=int, default=32, help="Batch size for data loading (default: 32)"
    )
    cl_parser.add_argument(
        "--no_whitening",
        action="store",
        type=bool,
        nargs="?",
        required=False,
        const=True,
        default=False,
    )
    args = cl_parser.parse_args()

    classifier_inference(
        folder_with_mrcs=args.folder_with_mrcs,
        estimator_weights=args.estimator_weights,
        estimator_config=args.estimator_config,
        file_name=args.file_name,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        max_batch_size=args.max_batch_size,
        whitening=args.no_whitening is False,
    )