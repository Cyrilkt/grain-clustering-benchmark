from __future__ import print_function
import os.path as osp
import torch
from torchvision import transforms
import numpy as np
import torch.distributed as dist
from utils import TwoCropTransform, extract_features, concat_all_gather
from utils.ops import is_root_worker, dataset_with_indices
from utils.loggerx import LoggerX
import torch_clustering
from grain_benchmark.final_export import write_final_artifacts
from supervisor import ClusterAnalysis
from torch.utils.data import DataLoader
from PIL import Image
import torchvision

class TrainTask(object):
    single_view = False
    l2_normalize = True

    def __init__(self, opt):
        self.opt = opt
        if opt.dataset != 'custom_big' or not opt.whole_dataset:
            raise ValueError('Grain image training requires custom_big and whole_dataset=true')
        if opt.num_cluster is None or int(opt.num_cluster) < 2:
            raise ValueError('An explicit positive initial K is required; no label-count fallback')
        self.verbose = is_root_worker()
        total_batch_size = opt.batch_size * opt.acc_grd_step
        if dist.is_initialized():
            total_batch_size *= dist.get_world_size()
        opt.learning_rate = opt.learning_rate * (total_batch_size / 256)
        self.cur_epoch = 1
        self.logger = LoggerX(save_root=osp.join(opt.output_root, opt.run_name))
        self.feature_extractor = None
        self.feature_extractor_copy = None
        self.set_loader()
        self.set_model()
        self.mem_features = None
        self.mem_labels = None
        self.last_partition_epoch = None

    @staticmethod
    def default_options():
        return dict(dataset='custom_big', whole_dataset=True, acc_grd_step=1, resume_epoch=0, amp=False, use_copy=False, pin_memory=True, num_workers=8, v2=True, queue_size=0, syncbn=False, fix_predictor_lr=False, exclude_bias_and_bn=False, num_devices=4, resized_crop_scale=0.08)

    @staticmethod
    def create_dataset(data_root, dataset_name, train, transform=None, memory=False, label_file=None, unlabeled=False):
        import json
        has_subfolder = False
        if dataset_name == 'custom_big':
            if not isinstance(data_root, dict):
                try:
                    data_root = json.loads(data_root)
                except Exception as e:
                    raise ValueError("For 'custom_big' dataset, opt.data_folder must be a dictionary with keys 'labeled' and 'unlabeled'.") from e

            def is_valid_image(file_path):
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        img.verify()
                    return file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp'))
                except Exception as e:
                    print(f'Skipping corrupt image: {file_path}')
                    return False
            if unlabeled:
                unlabeled_paths = data_root.get('unlabeled', None)
                if unlabeled_paths is None:
                    raise ValueError("For custom_big dataset, 'unlabeled' key must be present in data_folder")
                if isinstance(unlabeled_paths, list):
                    datasets_list = []
                    for path in unlabeled_paths:
                        print(f'Loading custom big unlabeled dataset from: {path}')
                        ds = torchvision.datasets.ImageFolder(root=path, transform=transform, is_valid_file=is_valid_image)
                        datasets_list.append(ds)
                    dataset = torch.utils.data.ConcatDataset(datasets_list)
                    has_subfolder = False
                else:
                    dataset_path = unlabeled_paths
                    print(f'Loading custom big unlabeled dataset from: {dataset_path}')
                    dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform, is_valid_file=is_valid_image)
                    has_subfolder = False
            else:
                dataset_path = data_root.get('labeled', None)
                if dataset_path is None:
                    raise ValueError("For custom_big dataset, 'labeled' key must be present in data_folder")
                print(f'Loading custom big labeled dataset from: {dataset_path}')
                dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform, is_valid_file=is_valid_image)
                has_subfolder = True
        else:
            raise ValueError("This benchmark supports only dataset='custom_big'.")
        return (dataset, has_subfolder)

    def build_dataloader(self, dataset_name, transform, batch_size, shuffle=False, drop_last=False, sampler=False, train=True, memory=False, data_resample=False, label_file=None, unlabeled=False):
        opt = self.opt
        dataset, _ = self.create_dataset(opt.data_folder, dataset_name, train, transform=transform, memory=memory, label_file=label_file, unlabeled=unlabeled)
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            labels = np.asarray([y for part in dataset.datasets for y in part.targets])
        else:
            labels = np.asarray(dataset.targets)
        if train and (not memory):
            dataset = dataset_with_indices(dataset)
        if sampler and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        else:
            sampler = None
        loader_kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=False if sampler is not None else shuffle, num_workers=opt.num_workers, pin_memory=opt.pin_memory, sampler=sampler, drop_last=drop_last)
        if opt.num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        return (torch.utils.data.DataLoader(**loader_kwargs), labels, sampler)

    def train_transform(self, normalize):
        opt = self.opt
        operations = [transforms.RandomResizedCrop(opt.img_size, scale=(opt.resized_crop_scale, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), transforms.RandomGrayscale(p=0.2)]
        if opt.use_gaussian_blur:
            operations.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5))
        operations.extend([transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), normalize])
        return TwoCropTransform(transforms.Compose(operations))

    def normalize(self, dataset_name):
        return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True)

    def export_final_artifacts(self):
        """Export the final encoder state, never an epoch selected with class labels.

        Re-extract AFTER the last optimizer step; cached pseudo-label features precede
        the final Grain update in the native fit loop and are not the final embedding.
        """
        from torch.utils.data import DataLoader
        if dist.is_initialized():
            dist.barrier()
        if is_root_worker():
            dataset = self.memory_loader.dataset
            loader = DataLoader(dataset, batch_size=self.opt.batch_size, shuffle=False, drop_last=False, num_workers=0)
            encoder = self.feature_extractor
            encoder.eval()
            features, labels = ([], [])
            with torch.no_grad():
                for images, target in loader:
                    device = next(encoder.parameters()).device
                    z = encoder(images.to(device)).float()
                    if self.l2_normalize:
                        z = torch.nn.functional.normalize(z, dim=1)
                    features.append(z.cpu())
                    labels.append(torch.as_tensor(target).cpu())
            features = torch.cat(features).numpy()
            labels = torch.cat(labels).numpy()
            write_final_artifacts(osp.join(self.opt.output_root, self.opt.run_name, 'final'), features, labels, epoch=int(self.opt.epochs), row_ids=np.arange(len(features), dtype=np.int64), metadata={'initial_K': int(self.opt.num_cluster), 'last_supervisor_K': int(self.num_cluster), 'method': self.opt.method, 'seed': int(self.opt.base_seed), 'input_manifest': 'image_paths.json'})
            from grain_benchmark.evaluation import clustering_metrics
            from grain_benchmark.cli import write_json
            from pathlib import Path
            run_root = Path(self.opt.output_root) / self.opt.run_name
            sample_ids = np.asarray([str(Path(p).relative_to(dataset.root)) for p, _ in dataset.samples])
            from grain_benchmark.predictions import image_dataset_digest
            dataset_digest = image_dataset_digest(sample_ids, labels)
            if self.last_partition_epoch is not None:
                predictions = self.psedo_labels.detach().cpu().numpy()
                partition_labels = self.mem_labels.detach().cpu().numpy()
                np.savez_compressed(run_root / 'predictions.npz', predictions=predictions, row_ids=np.arange(len(predictions), dtype=np.int64), labels=partition_labels, regime=np.asarray(self.opt.regime), partition_epoch=np.asarray(self.last_partition_epoch), sample_ids=sample_ids, dataset_sha256=np.asarray(dataset_digest), input_kind=np.asarray('images'))
                result = clustering_metrics(partition_labels, predictions)
                excluded = {'seed', 'base_seed', 'run_name', 'output_root', 'debug_dir'}
                result['config'] = {key: value for key, value in vars(self.opt).items() if key not in excluded}
                result.update(method=self.opt.method, regime=self.opt.regime, seed=self.opt.base_seed, input_kind='images', dataset_sha256=dataset_digest, partition_epoch=self.last_partition_epoch, selection='last_native_pseudolabel_update_before_final_grain_pass')
                write_json(run_root / 'metrics.json', result)
            import json
            paths = sample_ids.tolist()
            if len(paths) == len(features):
                with open(osp.join(self.opt.output_root, self.opt.run_name, 'final', 'image_paths.json'), 'w') as f:
                    json.dump(paths, f, indent=2)
        if dist.is_initialized():
            dist.barrier()

    def set_model(opt):
        pass

    def get_new_k(self, split=0.9, index=0):
        opt = self.opt
        if opt.use_copy:
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor
        mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
        if self.l2_normalize:
            mem_features = torch.nn.functional.normalize(mem_features, p=2, dim=1)
        gathered_features = concat_all_gather(mem_features)
        new_k_tensor = torch.tensor([0]).cuda()
        if torch.distributed.get_rank() == 0:
            total_samples = gathered_features.shape[0]
            num_samples = int(total_samples * split) if 0 < split < 1 else int(split)
            sampled_indices = torch.randint(0, total_samples, (num_samples,))
            sampled_features = gathered_features[sampled_indices]
            cluster_analysis_instance = ClusterAnalysis(num_models=opt.number_autoencoder, input_dim=sampled_features.size(1), latent_dim=opt.autoencoder_latent_dim, max_lr=0.01, features=sampled_features, num_epochs=opt.autoencoder_training_epoch, index=index, cluster_range=opt.cluster_range, rank=dist.get_rank(), n_candidate_autoencoder_clustering=opt.n_candidate_autoencoder_clustering, n_candidate_k_mean=opt.n_candidate_k_mean, n_candidate_k_max=opt.n_candidate_k_max, kmeans_n_init=opt.supervisor_kmeans_n_init)
            new_k, list_results, models_tracking_clusterization = cluster_analysis_instance.get_k()
            new_k_tensor += new_k
            torch.save(list_results, osp.join(opt.debug_dir, f'supervisor_candidates_epoch_{self.cur_epoch}.pth'))
            torch.save(models_tracking_clusterization, osp.join(opt.debug_dir, f'supervisor_partitions_epoch_{self.cur_epoch}.pth'))
        dist.barrier()
        torch.distributed.broadcast(new_k_tensor, src=0)
        self.num_cluster = new_k_tensor.item()

    def cosine_annealing_LR(self, n_iter):
        opt = self.opt
        epoch = n_iter / self.iter_per_epoch
        max_lr = opt.learning_rate
        min_lr = max_lr * opt.learning_eta_min
        if epoch < opt.warmup_epochs:
            lr = opt.learning_rate * epoch / opt.warmup_epochs
        else:
            lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos((epoch - opt.warmup_epochs) * np.pi / opt.epochs))
        return lr

    def clustering(self, features, n_clusters):
        opt = self.opt
        kwargs = {'metric': 'cosine' if self.l2_normalize else 'euclidean', 'distributed': True, 'random_state': 0, 'n_clusters': n_clusters, 'verbose': True}
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=0.0001, n_init=opt.kmeans_n_init, **kwargs)
        psedo_labels = clustering_model.fit_predict(features)
        cluster_centers = clustering_model.cluster_centers_
        return (psedo_labels, cluster_centers)

    def psedo_labeling(self, n_iter):
        opt = self.opt
        torch.cuda.empty_cache()
        if opt.use_copy:
            self.feature_extractor_copy.load_state_dict(self.feature_extractor.state_dict())
            for value in self.feature_extractor_copy.state_dict().values():
                dist.broadcast(value, src=0)
            feature_extractor = self.feature_extractor_copy
        else:
            feature_extractor = self.feature_extractor
        mem_features, mem_labels = extract_features(feature_extractor, self.memory_loader)
        if self.l2_normalize:
            mem_features.div_(torch.linalg.norm(mem_features, dim=1, ord=2, keepdim=True))
        psedo_labels, cluster_centers = self.clustering(mem_features, self.num_cluster)
        dist.barrier()
        dist.broadcast(psedo_labels, src=0)
        dist.broadcast(cluster_centers, src=0)
        self.psedo_labels.copy_(psedo_labels)
        self.cluster_centers = cluster_centers
        self.mem_features, self.mem_labels = (mem_features, mem_labels)
        self.last_partition_epoch = self.cur_epoch
        from grain_benchmark.evaluation import clustering_metrics
        result = clustering_metrics(mem_labels.cpu().numpy(), psedo_labels.cpu().numpy())
        self.logger.msg({key: result[key] for key in ('ACC', 'NMI', 'ARI', 'K_occupied')}, n_iter)
        torch.cuda.empty_cache()

    def collect_params(self, *models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                param_dict = {'name': name, 'params': param}
                if exclude_bias_and_bn and any((s in name for s in ['bn', 'bias'])):
                    param_dict.update({'weight_decay': 0.0, 'lars_exclude': True})
                param_list.append(param_dict)
        return param_list
