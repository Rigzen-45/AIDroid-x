"""
Genetic Algorithm Module (GAM) for Malware Detection
Paper params: population=10, step_size=40, generations=50
Uses graph encoding and evolutionary optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from deap import base, creator, tools, algorithms
import random
import logging
from torch_geometric.nn import GCNConv, global_mean_pool

logger = logging.getLogger(__name__)


class GraphEncoder(nn.Module):
    """
    Neural network for encoding graphs into fixed-size vectors.
    Used by genetic algorithm agents.
    """
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64, 
                 embedding_dim: int = 32):
        """
        Initialize graph encoder.
        
        Args:
            num_node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Output embedding dimension
        """
        super(GraphEncoder, self).__init__()
        
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embedding_dim)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x, edge_index, batch):
        """
        Encode graph to fixed-size embedding.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph embedding [batch_size, embedding_dim]
        """
        # GCN layers
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv3(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return x


class GAMClassifier(nn.Module):
    """
    Complete GAM classifier combining graph encoder and classification head.
    Individual agents in genetic algorithm have different weights.
    """
    
    def __init__(self, num_node_features: int, hidden_dim: int = 64,
                 embedding_dim: int = 32, num_classes: int = 2):
        """
        Initialize GAM classifier.
        
        Args:
            num_node_features: Input node feature dimension
            hidden_dim: Hidden layer dimension
            embedding_dim: Graph embedding dimension
            num_classes: Number of output classes
        """
        super(GAMClassifier, self).__init__()
        
        self.encoder = GraphEncoder(num_node_features, hidden_dim, embedding_dim)
        
        # Classification head
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.bn_fc2 = nn.BatchNorm1d(32)
        
        # Store attention-like scores for explainability
        self.node_importance = None
        
    def forward(self, x, edge_index, batch, return_embedding: bool = False):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            return_embedding: If True, return embedding before classification
            
        Returns:
            Logits [batch_size, num_classes] or (logits, embedding)
        """
        # Encode graph
        embedding = self.encoder(x, edge_index, batch)
        
        if return_embedding:
            return embedding
        
        # Classification
        out = self.fc1(embedding)
        out = self.bn_fc1(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        
        out = self.fc2(out)
        out = self.bn_fc2(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.5, training=self.training)
        
        logits = self.fc3(out)
        
        return logits
    
    def extract_node_importance(self, x, edge_index, batch):
        """
        Extract node importance scores for explainability.
        Uses gradient-based attribution.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            
        Returns:
            Node importance scores [num_nodes]
        """
        self.eval()
        x = x.requires_grad_(True)
        
        # Forward pass
        logits = self.forward(x, edge_index, batch)
        
        # Get prediction for malware class
        malware_score = logits[:, 1].sum()
        
        # Compute gradients
        malware_score.backward()
        
        # Node importance = gradient magnitude
        importance = torch.abs(x.grad).sum(dim=1)
        
        return importance.detach()
    
    def get_weights_as_vector(self) -> np.ndarray:
        """
        Get all model weights as a flat vector (for genetic algorithm).
        
        Returns:
            Numpy array of all parameters
        """
        weights = []
        for param in self.parameters():
            weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def set_weights_from_vector(self, vector: np.ndarray):
        """
        Set model weights from a flat vector (for genetic algorithm).
        
        Args:
            vector: Flat numpy array of parameters
        """
        idx = 0
        for param in self.parameters():
            param_length = param.numel()
            param_shape = param.shape
            
            # Extract slice and reshape
            param_data = vector[idx:idx + param_length].reshape(param_shape)
            param.data = torch.from_numpy(param_data).float().to(param.device)
            
            idx += param_length


class GeneticAlgorithmTrainer:
    """
    Genetic Algorithm trainer for GAM model.
    Evolves population of neural networks using evolutionary strategies.
    """
    
    def __init__(
        self,
        num_node_features: int,
        population_size: int = 10,
        step_size: int = 40,
        num_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        tournament_size: int = 3,
        elitism: int = 2,
        device: str = "cuda"
    ):
        """
        Initialize genetic algorithm trainer.
        
        Args:
            num_node_features: Input feature dimension
            population_size: Number of agents in population
            step_size: Training steps per generation
            num_generations: Total number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            tournament_size: Size for tournament selection
            elitism: Number of top agents to preserve
            device: Device to train on
        """
        self.num_node_features = num_node_features
        self.population_size = population_size
        self.step_size = step_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.device = device
        
        # Initialize population
        self.population = self._initialize_population()
        
        # DEAP setup for genetic operations
        self._setup_deap()
        
        # Best model tracking
        self.best_agent = None
        self.best_fitness = -float('inf')
        self.fitness_history = []
        
        logger.info(f"Initialized GAM with population={population_size}, "
                   f"step_size={step_size}, generations={num_generations}")
    
    def _initialize_population(self) -> List[GAMClassifier]:
        """
        Initialize population of neural networks with random weights.
        
        Returns:
            List of GAMClassifier models
        """
        population = []
        for i in range(self.population_size):
            model = GAMClassifier(
                num_node_features=self.num_node_features,
                hidden_dim=64,
                embedding_dim=32,
                num_classes=2
            ).to(self.device)
            population.append(model)
        
        logger.info(f"Initialized population of {self.population_size} agents")
        return population
    
    def _setup_deap(self):
        """Setup DEAP framework for genetic operations."""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        
        # Register genetic operations
        self.toolbox.register("select", tools.selTournament, 
                            tournsize=self.tournament_size)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
    
    def _evaluate_agent(self, agent: GAMClassifier, data_loader) -> float:
        """
        Evaluate fitness of a single agent.
        
        Args:
            agent: GAMClassifier model
            data_loader: PyG DataLoader
            
        Returns:
            Fitness score (accuracy)
        """
        agent.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                logits = agent(batch.x, batch.edge_index, batch.batch)
                preds = torch.argmax(logits, dim=1)
                
                correct += (preds == batch.y).sum().item()
                total += batch.y.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def _train_agent(self, agent: GAMClassifier, train_loader, 
                    criterion, optimizer, steps: int):
        """
        Train single agent for given number of steps.
        
        Args:
            agent: GAMClassifier model
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            steps: Number of training steps
        """
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
                optimizer.step()
                
                step_count += 1
    
    def _crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        Crossover operation: combine weights from two parents.
        
        Args:
            ind1: First parent (weight vector)
            ind2: Second parent (weight vector)
            
        Returns:
            Two offspring
        """
        if random.random() < self.crossover_rate:
            # Single-point crossover
            point = random.randint(1, len(ind1) - 1)
            ind1[point:], ind2[point:] = ind2[point:], ind1[point:]
        
        return ind1, ind2
    
    def _mutate(self, individual: List) -> Tuple[List]:
        """
        Mutation operation: randomly perturb weights.
        
        Args:
            individual: Weight vector
            
        Returns:
            Mutated individual
        """
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                # Gaussian mutation
                individual[i] += np.random.normal(0, 0.1)
        
        return individual,
    
    def train(self, train_loader, val_loader) -> GAMClassifier:
        """
        Train GAM using genetic algorithm.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Best trained model
        """
        logger.info("Starting GAM evolutionary training...")
        
        criterion = nn.CrossEntropyLoss()
        
        for generation in range(self.num_generations):
            logger.info(f"\n=== Generation {generation + 1}/{self.num_generations} ===")
            
            # Step 1: Train each agent for step_size iterations
            for i, agent in enumerate(self.population):
                optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
                self._train_agent(agent, train_loader, criterion, 
                                optimizer, self.step_size)
            
            # Step 2: Evaluate fitness of each agent
            fitness_scores = []
            for i, agent in enumerate(self.population):
                fitness = self._evaluate_agent(agent, val_loader)
                fitness_scores.append(fitness)
                logger.info(f"  Agent {i}: Fitness = {fitness:.4f}")
            
            # Track best agent
            max_fitness = max(fitness_scores)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                best_idx = fitness_scores.index(max_fitness)
                self.best_agent = self.population[best_idx]
                logger.info(f"  *** New best fitness: {max_fitness:.4f} ***")
            
            self.fitness_history.append(max_fitness)
            
            # Step 3: Selection and reproduction
            if generation < self.num_generations - 1:
                self.population = self._evolve_population(fitness_scores)
        
        logger.info(f"\nTraining complete. Best fitness: {self.best_fitness:.4f}")
        return self.best_agent
    
    def _evolve_population(self, fitness_scores: List[float]) -> List[GAMClassifier]:
        """
        Evolve population using genetic operations.
        
        Args:
            fitness_scores: Fitness of each agent
            
        Returns:
            New population
        """
        # Convert models to weight vectors
        population_vectors = []
        for agent in self.population:
            vector = agent.get_weights_as_vector()
            individual = creator.Individual(vector.tolist())
            individual.fitness.values = (fitness_scores[len(population_vectors)],)
            population_vectors.append(individual)
        
        # Elitism: preserve best agents
        elite_indices = np.argsort(fitness_scores)[-self.elitism:]
        elite_agents = [self.population[i] for i in elite_indices]
        
        # Select parents
        offspring = self.toolbox.select(population_vectors, 
                                       self.population_size - self.elitism)
        offspring = list(map(self.toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            self.toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            self.toolbox.mutate(mutant)
            del mutant.fitness.values
        
        # Create new population
        new_population = elite_agents.copy()
        
        for individual in offspring:
            new_agent = GAMClassifier(
                num_node_features=self.num_node_features,
                hidden_dim=64,
                embedding_dim=32,
                num_classes=2
            ).to(self.device)
            new_agent.set_weights_from_vector(np.array(individual))
            new_population.append(new_agent)
        
        return new_population


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test GAM model
    model = GAMClassifier(num_node_features=15)
    
    print(f"GAM Model:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test forward pass
    batch_size = 2
    num_nodes = 50
    num_edges = 100
    
    x = torch.randn(num_nodes, 15)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.cat([torch.zeros(25, dtype=torch.long),
                      torch.ones(25, dtype=torch.long)])
    
    logits = model(x, edge_index, batch)
    print(f"\nTest forward pass:")
    print(f"  Output logits shape: {logits.shape}")
    
    # Test weight vector conversion
    weights = model.get_weights_as_vector()
    print(f"  Weight vector size: {weights.shape}")
    
    model.set_weights_from_vector(weights)
    logits2 = model(x, edge_index, batch)
    print(f"  Weights restored correctly: {torch.allclose(logits, logits2)}")