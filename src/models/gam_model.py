"""
Genetic Algorithm Module (GAM) - FIXED VERSION
Critical fixes:
1. Class-weighted loss
2. Gradient clipping
3. Better learning rate
4. Proper dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from deap import base, creator, tools
import random
import logging
import copy
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv,global_mean_pool, global_max_pool

logger = logging.getLogger(__name__)


class GraphEncoder(nn.Module):
    """Graph encoder for GAM."""
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64, embedding_dim: int = 32):
        super(GraphEncoder, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)  
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        x = torch.cat([x_mean, x_max], dim=1)
        
        return x


class GAMClassifier(nn.Module):
    """GAM classifier with proper regularization."""
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64,
                 embedding_dim: int = 32, num_classes: int = 2):
        super(GAMClassifier, self).__init__()
        
        self.encoder = GraphEncoder(num_node_features, hidden_dim, embedding_dim)
        
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.bn_fc2 = nn.BatchNorm1d(32)
        
    def forward(self, x, edge_index, batch, return_embedding: bool = False):
        embedding = self.encoder(x, edge_index, batch)
        
        if return_embedding:
            return embedding
        
        out = self.fc1(embedding)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)
        
        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)
        
        logits = self.fc3(out)
        return logits
    
    def extract_node_importance(self, x, edge_index, batch):
        self.eval()

        x_inp = x.detach().clone().requires_grad_(True)

        logits = self.forward(x_inp, edge_index, batch)
        malware_score = logits[:, 1].sum()
        malware_score.backward()

        importance = torch.abs(x_inp.grad).sum(dim=1)

        return importance.detach()
    
    def get_weights_as_vector(self) -> np.ndarray:
        """Get all weights as flat vector."""
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, vector: np.ndarray):
        """Set weights from flat vector."""
        idx = 0
        for param in self.parameters():
            param_length = param.numel()
            param_shape = param.shape
            param_data = vector[idx:idx + param_length].reshape(param_shape)
            param.data = torch.from_numpy(param_data).float().to(param.device)
            idx += param_length


class GeneticAlgorithmTrainer:
    """Genetic Algorithm trainer with class weighting."""

    def __init__(
        self,
        num_node_features: int,
        population_size: int = 20,
        step_size: int = 80,
        num_generations: int = 30,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 2,
        device: str = "cuda",
        class_weights: Optional[torch.Tensor] = None
    ):
        self.num_node_features = num_node_features
        self.population_size = population_size
        self.step_size = step_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.device = device
        self.class_weights = class_weights

        self.population = self._initialize_population()
        self._setup_deap()

        self.best_agent = None
        self.best_fitness = -float("inf")
        self.fitness_history = []

        logger.info(
            f"GAM initialized: pop={population_size}, step={step_size}, gen={num_generations}"
        )

    # --------------------------------------------------

    def _initialize_population(self):
        population = []
        self.optimizers = []

        for _ in range(self.population_size):
            model = GAMClassifier(
                num_node_features=self.num_node_features
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

            population.append(model)
            self.optimizers.append(optimizer)

        return population

    # --------------------------------------------------

    def _setup_deap(self):
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("clone", copy.deepcopy)

    # --------------------------------------------------

    def _evaluate_agent(self, agent: GAMClassifier, data_loader) -> float:
        agent.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)

                logits = agent(batch.x, batch.edge_index, batch.batch)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(batch.y.cpu().numpy().flatten())

        return f1_score(all_labels, all_preds, average="binary", zero_division=0)

    # --------------------------------------------------

    def _train_agent(self, agent, train_loader, criterion, optimizer, steps):
        agent.train()
        step_count = 0

        while step_count < steps:
            for batch in train_loader:
                if step_count >= steps:
                    break

                batch = batch.to(self.device)

                optimizer.zero_grad()
                logits = agent(batch.x, batch.edge_index, batch.batch)
                loss = criterion(logits, batch.y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
                optimizer.step()

                step_count += 1

    # --------------------------------------------------

    def train(self, train_loader, val_loader, writer=None) -> GAMClassifier:
        """
        Run the full evolutionary training loop.

        Args:
            train_loader: PyG DataLoader for training set.
            val_loader:   PyG DataLoader for validation set.
            writer:       Optional SummaryWriter.  When provided, fitness
                          scalars are written live after EVERY generation so
                          TensorBoard shows real-time progress during the
                          multi-hour GA run.  [L-02 FIX]
        """
        if self.class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device)
            )
        else:
            criterion = nn.CrossEntropyLoss()

        for generation in range(self.num_generations):

            logger.info(
                f"\n=== Generation {generation + 1}/{self.num_generations} ==="
            )

            # 1️⃣ Train
            for i, agent in enumerate(self.population):
                self._train_agent(
                    agent,
                    train_loader,
                    criterion,
                    self.optimizers[i],
                    self.step_size
                )

            # 2️⃣ Evaluate
            fitness_scores = []
            for i, agent in enumerate(self.population):
                fitness = self._evaluate_agent(agent, val_loader)
                fitness_scores.append(fitness)
                logger.info(f"Agent {i}: Fitness = {fitness:.4f}")

            max_fitness = max(fitness_scores)
            mean_fitness = float(np.mean(fitness_scores))

            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                self.best_agent = copy.deepcopy(self.population[best_idx])
                logger.info(f" New best: {max_fitness:.4f}")

            self.fitness_history.append(max_fitness)

            # [L-02 FIX] Write live per-generation scalars so TensorBoard
            # shows progress during training, not only after it completes.
            if writer is not None:
                writer.add_scalar("Fitness/best",  max_fitness,  generation)
                writer.add_scalar("Fitness/mean",  mean_fitness, generation)

            # 3️⃣ Evolve
            if generation < self.num_generations - 1:
                self.population = self._evolve_population(fitness_scores)

        logger.info(f"\nTraining complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_agent

    # --------------------------------------------------

    def _evolve_population(self, fitness_scores: List[float]):

        population_vectors = []

        for i, agent in enumerate(self.population):
            vector = agent.get_weights_as_vector()
            individual = creator.Individual(vector.tolist())
            individual.fitness.values = (fitness_scores[i],)
            population_vectors.append(individual)

        # Elites (deepcopy to protect them)
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        elite_agents = [
            copy.deepcopy(self.population[i]) for i in elite_indices
        ]

        offspring = self.toolbox.select(
            population_vectors,
            self.population_size - self.elitism
        )
        offspring = list(map(self.toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.crossover_rate:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            self.toolbox.mutate(mutant)
            del mutant.fitness.values

        new_population = elite_agents.copy()
        new_optimizers = []

        # New optimizers for elites
        for agent in elite_agents:
            new_optimizers.append(
                torch.optim.Adam(agent.parameters(), lr=0.0005)
            )

        # Create mutated agents
        for individual in offspring:
            new_agent = GAMClassifier(
                num_node_features=self.num_node_features
            ).to(self.device)

            new_agent.set_weights_from_vector(np.array(individual))
            new_population.append(new_agent)

            new_optimizers.append(
                torch.optim.Adam(new_agent.parameters(), lr=0.0005)
            )

        self.optimizers = new_optimizers

        return new_population

    # --------------------------------------------------

    def _crossover(self, ind1: List, ind2: List):
        point = random.randint(1, len(ind1) - 1)
        ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        return ind1, ind2

    def _mutate(self, individual: List):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        return individual,
