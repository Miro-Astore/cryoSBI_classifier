import os
import time
import logging
import torch
from torchvision import transforms
import cryo_sbi.utils.image_utils as img_utils
import cryo_sbi.utils.classifier_utils as cls_utils


torch.backends.cudnn.benchmark = True


def setup_logging(debug: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def get_file_list(folder_with_mrcs):
    """
    Get a sorted list of .mrc file paths from the specified folder.

    Args:
        folder_with_mrcs (str): Path to the folder containing .mrc files.

    Returns:
        List[str]: Sorted list of .mrc file paths.
    """

    particle_paths = [os.path.join(folder_with_mrcs, f) for f in os.listdir(folder_with_mrcs) if f.endswith(".mrc") or f.endswith(".mrcs")]
    try:
        particle_paths = sorted(particle_paths, key=lambda x: int(os.path.basename(x).split("_")[1]))
    except:
        print("Could not sort particle paths by number, sorting alphabetically instead.")
    return particle_paths


def classifier_inference(
        folder_with_mrcs, 
        estimator_weights, 
        estimator_config, 
        file_name, 
        num_workers, 
        output_dir, 
        image_size, 
        prefetch_factor, 
        max_batch_size,
        whitening: bool = True,
    ):
    setup_logging()

    assert os.path.exists(folder_with_mrcs), f"Folder {folder_with_mrcs} does not exist."
    assert os.path.exists(estimator_weights), f"Estimator weights {estimator_weights} do not exist."
    assert os.path.exists(estimator_config), f"Estimator config {estimator_config} does not exist."
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist."
    assert torch.cuda.is_available(), "CUDA is not available."
    
    transform = transforms.Compose([
        img_utils.WhitenImage(image_size) if whitening else img_utils.Identity(),
        img_utils.NormalizeIndividual(),
    ])

    particle_paths = get_file_list(folder_with_mrcs)
    logging.info(f"Found {len(particle_paths)} .mrc files.")
    logging.info(f"Analyzing mrc files :\n" + "\n".join([p.split('/')[-1] for p in particle_paths]) + "\n")


    classifier = cls_utils.load_classifier(
        estimator_config,
        estimator_weights,
        device="cuda",
    )
    classifier.eval()

    loader = img_utils.MRCloader(
        particle_paths,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        in_order=False,
    )

    results = []
    start_time = time.time()

    with torch.inference_mode():
        for idx, images in loader:
                if images.shape[0] > max_batch_size:
                    logits_batched, embeddings_batched = [], []
                    for image_batch in torch.split(images.cuda(non_blocking=True), split_size_or_sections=max_batch_size, dim=0):
                        transformed_images = transform(image_batch)
                        logits, embeddings = classifier.logits_embedding(-transformed_images)
                        logits_batched.append(logits)
                        embeddings_batched.append(embeddings)
                    logits = torch.cat(logits_batched, dim=0)
                    embeddings = torch.cat(embeddings_batched, dim=0)
                    results.append((idx, logits.cpu(), embeddings.cpu()))
                else:
                    transformed_images = transform(images.cuda(non_blocking=True))
                    logits, embeddings = classifier.logits_embedding(-transformed_images)
                    results.append((idx, logits.cpu(), embeddings.cpu()))

    end_time = time.time()
    duration = end_time - start_time

    results = sorted(results, key=lambda x: x[0])
    likelihoods = torch.cat([r[1] for r in results], dim=0)
    embeddings = torch.cat([r[2] for r in results], dim=0)

    torch.save(likelihoods, os.path.join(output_dir, f"likelihoods_{file_name}"))
    torch.save(embeddings, os.path.join(output_dir, f"embeddings_{file_name}"))
    logging.info(f"Inference completed in {duration:.2f} seconds for {likelihoods.shape[0]} images.")
