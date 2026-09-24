import torch_clustering
import torch
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

def silhouette_scores_cosine_distance(X, labels):
    """
    Computes the silhouette scores using cosine distance for each sample in the dataset using PyTorch.
    Args:
    - X: A PyTorch tensor of shape (n_samples, n_features) representing the dataset.
    - labels: A PyTorch tensor of shape (n_samples,) representing the cluster labels for each sample.
    Returns:
    - silhouette_scores: A PyTorch tensor of shape (n_samples,) containing the silhouette score for each sample.
    """
    X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
    cosine_sim = torch.mm(X_norm, X_norm.t())
    cosine_dist = 1 - cosine_sim
    a = torch.zeros(X.size(0), device=X.device)
    b = torch.full((X.size(0),), float('inf'), device=X.device)
    unique_labels = labels.unique()
    for label in unique_labels:
        in_cluster = labels == label
        out_cluster = labels != label
        a[in_cluster] = cosine_dist[in_cluster][:, in_cluster].sum(dim=1) / (in_cluster.sum() - 1).clamp(min=1)
        for other_label in unique_labels:
            if label == other_label:
                continue
            other_cluster = labels == other_label
            dist_to_other_cluster = cosine_dist[in_cluster][:, other_cluster].mean(dim=1)
            b[in_cluster] = torch.min(b[in_cluster], dist_to_other_cluster)
    s = (b - a) / torch.max(a, b)
    s[torch.isnan(s)] = 0
    return s

def average_difference_to_total_avg(silhouette_scores, total_avg):
    differences = silhouette_scores - total_avg
    avg_difference = torch.mean(differences)
    return avg_difference.item()

def silhouette_scores_by_cluster(X, cluster_labels):
    is_pytorch = isinstance(X, torch.Tensor)
    silhouette_scores = silhouette_scores_cosine_distance(X, cluster_labels)
    if is_pytorch:
        total_avg_silhouette_score = torch.mean(silhouette_scores).item()
        unique_labels = torch.unique(cluster_labels)
    else:
        total_avg_silhouette_score = np.mean(silhouette_scores)
        unique_labels = np.unique(cluster_labels)
    scores_by_cluster = {}
    for label in unique_labels:
        if is_pytorch:
            scores = silhouette_scores[cluster_labels == label]
            sorted_scores, _ = torch.sort(scores, descending=True)
            scores_by_cluster[label.item()] = sorted_scores
        else:
            scores = silhouette_scores[cluster_labels == label]
            scores_by_cluster[label] = np.sort(scores)[::-1]
    return (scores_by_cluster, total_avg_silhouette_score)

def jensen_shannon_divergence(scores_by_cluster, device, epsilon=1e-10):
    max_length = max((len(scores) for scores in scores_by_cluster.values()))
    min_score = min([scores.min() for scores in scores_by_cluster.values()])
    shift_value = abs(min_score) + epsilon
    distributions = []
    for cluster_id, scores in scores_by_cluster.items():
        shifted_scores = scores + shift_value
        if len(shifted_scores) < max_length:
            padded_scores = torch.cat((shifted_scores, torch.zeros(max_length - len(shifted_scores), device=device)))
        else:
            padded_scores = shifted_scores
        normalized_scores = padded_scores / padded_scores.sum()
        distributions.append(normalized_scores)
    distributions = torch.stack(distributions)
    weights = torch.tensor([len(scores) for scores in scores_by_cluster.values()], dtype=torch.float32, device=device)
    weights /= weights.sum()
    mixture = torch.zeros(max_length, device=device)
    for dist, weight in zip(distributions, weights):
        mixture += dist * weight
    weighted_entropies = torch.sum(weights * -torch.sum(distributions * torch.log(distributions + 1e-10), dim=1))
    mixture_entropy = -torch.sum(mixture * torch.log(mixture + 1e-10))
    jsd = mixture_entropy - weighted_entropies
    return jsd.item()

def cluster_analysis(X, cluster_range, k_select, n_select, kmeans_n_init=10):
    results = []
    best_labels = None
    best_k = None
    for k in cluster_range:
        clustering_model = torch_clustering.PyTorchKMeans(init='k-means++', max_iter=300, tol=0.0001, n_clusters=k, metric='cosine', n_init=kmeans_n_init)
        labels = clustering_model.fit_predict(X)
        silhouette_scores = silhouette_scores_cosine_distance(X, labels)
        total_avg_silhouette_score = torch.mean(silhouette_scores)
        avg_diff = average_difference_to_total_avg(silhouette_scores, total_avg_silhouette_score)
        scores_by_cluster, _ = silhouette_scores_by_cluster(X, labels)
        jsd = jensen_shannon_divergence(scores_by_cluster, X.device)
        results.append((k, total_avg_silhouette_score, avg_diff, jsd, labels))
    optimal_cluster_info = select_optimal_clusters_v3(results, n_select)
    best_k, _, _, _, best_labels = optimal_cluster_info
    return (best_k, best_labels, optimal_cluster_info, results)

def select_optimal_clusters_v3(results, k_select):
    sorted_by_total_avg = sorted(results, key=lambda x: x[1], reverse=True)
    top_k_by_total_avg = sorted_by_total_avg[:k_select]
    optimal_result = min(top_k_by_total_avg, key=lambda x: x[3])
    return optimal_result

class Autoencoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.4), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(inplace=True), nn.Linear(16, 32), nn.ReLU(inplace=True), nn.Linear(32, 64), nn.ReLU(inplace=True), nn.Linear(64, 128), nn.ReLU(inplace=True), nn.Linear(128, input_dim))

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return (x_recon, latent)

    @staticmethod
    def cosine_dissimlarity_loss(output, target):
        normalized_output = nn.functional.normalize(output, dim=1)
        normalized_target = nn.functional.normalize(target, dim=1)
        loss = 1 - torch.sum(normalized_output * normalized_target, dim=1)
        return loss.mean()

    @staticmethod
    def generate_masked_inputs(inputs, mask_ratio=0.8):
        mask = torch.rand(inputs.shape) < mask_ratio
        mask = mask.to(inputs.device)
        masked_inputs = inputs * mask.float()
        return masked_inputs

class AutoencoderManager:

    def __init__(self, num_models, input_dim, latent_dim, rank):
        """
        Initializes the AutoencoderManager with a specified number of Autoencoder models.
        """
        self.rank = rank
        self.models = [Autoencoder(input_dim, latent_dim).cuda(self.rank) for _ in range(num_models)]
        self.optimizers = None
        self.schedulers = None
        self.labels_results = {}

    def train_models(self, data_loader, num_epochs=150, learning_rate=0.001, max_lr=0.01):
        """
        Train the supervisor autoencoders with cosine reconstruction and OneCycleLR.
        """
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.schedulers = [OneCycleLR(optimizer, max_lr=max_lr, total_steps=len(data_loader) * num_epochs) for optimizer in self.optimizers]
        for model_index, (model, optimizer, scheduler) in enumerate(zip(self.models, self.optimizers, self.schedulers)):
            print(f'Training model {model_index + 1}/{len(self.models)}')
            model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for inputs in data_loader:
                    inputs = inputs.cuda(self.rank)
                    inputs_masked = Autoencoder.generate_masked_inputs(inputs, mask_ratio=0.1)
                    optimizer.zero_grad()
                    outputs, _ = model(inputs_masked)
                    loss = Autoencoder.cosine_dissimlarity_loss(outputs, inputs)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    epoch_loss += loss.item()
                if self.rank == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Model {model_index + 1}, Loss: {epoch_loss / len(data_loader)}')

    def run_cluster_analysis(self, data_loader, cluster_range=range(2, 100), k_select=10, n_select=3, kmeans_n_init=10):
        """
        Runs cluster analysis on the latent spaces of all autoencoders managed by this AutoencoderManager.
        :param data_loader: DataLoader providing batches of input data.
        :param cluster_range: Range of cluster counts to consider for KMeans clustering.
        :param k_select: Top K selections based on average silhouette score.
        :param n_select: Select N from the top K based on the lowest average difference.
        """
        results = {}
        models_tracking_clusterization = {}
        for model_index, model in enumerate(self.models):
            model.eval().cuda(self.rank)
            all_latents = []
            with torch.no_grad():
                for batch in data_loader:
                    inputs = batch
                    _, latents = model(inputs.cuda(self.rank))
                    all_latents.append(latents)
            all_latents = torch.cat(all_latents, 0)
            best_k, best_labels, optimal_cluster_info, tracking_clusterization = cluster_analysis(all_latents, cluster_range, k_select, n_select, kmeans_n_init)
            models_tracking_clusterization[model_index] = tracking_clusterization
            results[model_index] = {'best_k': best_k, 'best_labels': best_labels, 'optimal_cluster_info': optimal_cluster_info}
        self.labels_results = results
        return (results, models_tracking_clusterization)

class ClusterAnalysis:

    def __init__(self, num_models, input_dim, latent_dim, max_lr, features, cluster_range, num_epochs, index, n_candidate_autoencoder_clustering, n_candidate_k_mean, n_candidate_k_max, rank, kmeans_n_init=10):
        self.num_models = num_models
        self.kmeans_n_init = kmeans_n_init
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.max_lr = max_lr
        self.num_epochs = num_epochs
        self.rank = rank
        self.rank = rank
        self.data_loader = DataLoader(features, batch_size=128, shuffle=True)
        self.index = index
        self.lower_range = cluster_range[0]
        self.upper_range = cluster_range[1]
        self.n_candidate_autoencoder_clustering = n_candidate_autoencoder_clustering
        self.n_candidate_k_mean = n_candidate_k_mean
        self.n_candidate_k_max = n_candidate_k_max

    def find_best_k_jensen_shannon(self, results, n_select=3):
        extracted_info = [(key, result['optimal_cluster_info'][3], result['best_k']) for key, result in results.items()]
        sorted_by_js_divergence = sorted(extracted_info, key=lambda x: x[1])
        top_n_by_js_divergence = sorted_by_js_divergence[:n_select]
        optimal_result = max(top_n_by_js_divergence, key=lambda x: x[2])
        optimal_key = optimal_result[0]
        optimal_cluster_dict = results[optimal_key]
        best_k = optimal_cluster_dict['best_k']
        return (best_k, best_k)

    def find_best_k_jensen_shannon_by_mean(self, results, n_select=3):
        extracted_info = [(key, result['optimal_cluster_info'][3], result['best_k']) for key, result in results.items()]
        sorted_by_js_divergence = sorted(extracted_info, key=lambda x: x[1])
        top_n_by_js_divergence = sorted_by_js_divergence[:n_select]
        mean_best_k = sum((x[2] for x in top_n_by_js_divergence)) / n_select
        rounded_mean_best_k = int(round(mean_best_k))
        return (rounded_mean_best_k, mean_best_k)

    def get_k(self):
        autoencoder_manager = AutoencoderManager(self.num_models, self.input_dim, self.latent_dim, self.rank)
        autoencoder_manager.train_models(self.data_loader, self.num_epochs, 0.001, self.max_lr)
        results, models_tracking_clusterization = autoencoder_manager.run_cluster_analysis(self.data_loader, cluster_range=range(self.lower_range, self.upper_range), k_select=10, n_select=self.n_candidate_autoencoder_clustering, kmeans_n_init=self.kmeans_n_init)
        print('Supervisor candidate K:', [r['best_k'] for r in results.values()])
        if self.index != 0:
            best_k, k = self.find_best_k_jensen_shannon_by_mean(results, n_select=self.n_candidate_k_mean)
        else:
            best_k, k = self.find_best_k_jensen_shannon(results, n_select=self.n_candidate_k_max)
        results['selected_k'] = best_k
        results['selected_k_not_rounded'] = k
        self.K = best_k
        return (best_k, results, models_tracking_clusterization)
